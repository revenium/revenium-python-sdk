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
- **Distributed Tracing**: Built-in trace visualization fields for cross-service observability
- **Asynchronous Processing**: Usage records are sent from a fixed pool of background threads, never on your request path
- **Graceful Shutdown**: Flushes pending metering data when the process exits, and when a `multiprocessing` worker process finishes its job, within a bounded time, without taking over your application's signal handling; `flush_metering()` flushes on demand
- **Thread-Safe**: Production-ready with `contextvars`-based context management for concurrent applications; wrap worker thread targets with `carry_context` (see [Decorator Support](#decorator-support))

## Supported Providers

| Provider | Extra | Install Command |
|----------|-------|----------------|
| OpenAI | `openai` | `pip install revenium-python-sdk[openai]` |
| Azure OpenAI | `openai` | `pip install revenium-python-sdk[openai]` |
| xAI (Grok, via OpenAI SDK) | `openai` | `pip install revenium-python-sdk[openai]` |
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

Not supported: Runway video generation is not metered automatically; to record those calls, wrap them with `@meter_tool` (see [Tool Metering](#tool-metering)).

## Feature Matrix

| Feature | OpenAI | Anthropic | Google | Ollama | LiteLLM | Perplexity | fal.ai |
|---------|--------|-----------|--------|--------|---------|------------|--------|
| Chat Completions | Yes\* | Yes | Yes | Yes | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Embeddings | Yes | - | Yes | Yes | Yes | - | - |
| Vision/Multimodal | Yes | Yes | Yes | - | Yes | - | Yes |
| Audio (speech, transcription, translation) | Yes | - | - | - | Proxy mode | - | Yes |
| Image Generation | - | - | Yes | - | - | - | Yes |
| Video Generation | - | - | Yes[^veo] | - | - | - | Yes |
| Prompt Capture | Yes | Yes | Yes | - | Client mode[^litellm-capture] | - | - |
| Azure / Bedrock | Azure | Bedrock | Vertex AI | - | All | - | - |
| LangChain Integration | Yes | - | - | - | - | - | - |
| Griptape Integration | Yes | Yes | - | Yes | Yes | - | - |
| CrewAI Integration | Yes | Yes | Yes | - | Yes | - | - |
| Proxy Mode | - | - | - | - | Yes | - | - |

[^litellm-capture]: LiteLLM prompt capture covers chat calls your own code makes (`completion`, `acompletion` and their streams). The LiteLLM gateway plugin (`ReveniumGuardrail`) does not capture prompts or replies.

\* OpenAI's `chat.completions` and `responses` calls are metered[^openai-background]; the legacy `completions.create` endpoint is not.

[^openai-background]: An OpenAI background response (`responses.create(background=True)`) is recorded when your code's `responses.retrieve(...)`, a resumed stream (`responses.retrieve(..., stream=True)` or `responses.stream(response_id=...)`) or `responses.cancel(...)` first sees the job finish, with the `usage_metadata` you gave at create. A background job you never retrieve, or retrieve only from a different process than the one that created it or after a restart, is not recorded. A cancelled background job is recorded when OpenAI reports usage for it on the first call that sees it cancelled, usually `responses.cancel(...)`; a cancelled job with no usage on that call is not recorded. A background stream cut off part-way and never resumed or retrieved is not recorded. Background jobs created through `beta.responses` are not metered. A background job created through `responses.with_streaming_response.create(..., background=True)` is not recorded, even when you retrieve it later. Reading a finished job only through `responses.with_streaming_response.retrieve(...)` does not record it; the job stays pending, and the next `responses.retrieve(...)` or `responses.with_raw_response.retrieve(...)` in the same process records it once.

[^veo]: A Veo video through `google.genai` (the Gemini API, or Vertex AI with `vertexai=True`) is recorded when your code's `client.operations.get(...)` (or `client.aio.operations.get(...)`) first sees the operation finish, billed at the requested clip length times the videos returned (Google's response carries no duration). A failed operation is recorded as a failed, zero-second video. When the request sets `webhook_config` or `pubsub_topic`, the video is recorded when you submit it. A video you never poll, or poll from a different process than the one that submitted it, is not recorded.

Some billable provider calls are not metered at all: see [Calls the SDK does not meter](#calls-the-sdk-does-not-meter).

## Calls the SDK does not meter

The calls below are billed by the provider (some, such as batch and fine-tuning jobs, when the job runs rather than when you call them), but the SDK records no usage for them: nothing reaches Revenium and nothing is logged. On every build a test compares this list with the SDK's inventory of the public calls of the OpenAI, Anthropic, Perplexity, Gemini, Ollama, LiteLLM and fal clients, boto3's Bedrock Runtime client, and the Vertex AI model, session, batch prediction, tuning and context-cache classes, on the provider versions the SDK is tested on. Calls reached another way, for example LangChain classes that call their own vendor SDK, are outside that check, except LangChain embeddings, listed by hand below; wrapping one with `wrap()` logs a warning. Sync and async forms are folded into one name, and the `with_raw_response` and `with_streaming_response` forms of a listed call are not metered either. To record any of these calls, wrap it with `@meter_tool` (see [Tool Metering](#tool-metering)).

<!-- not-metered:start -->
- **OpenAI** (`OpenAI` and `AsyncOpenAI` clients): `batches.create`, `beta.agents.sessions.create`, `beta.agents.sessions.events.create`, `beta.agents.sessions.events.stream`, `beta.agents.sessions.stream`, `beta.agents.sessions.update`, `beta.chatkit.sessions.create`, `beta.realtime.connect`, `beta.responses.compact`, `beta.responses.connect`, `beta.responses.create`, `beta.threads.create_and_run`, `beta.threads.create_and_run_poll`, `beta.threads.create_and_run_stream`, `beta.threads.runs.create`, `beta.threads.runs.create_and_poll`, `beta.threads.runs.create_and_stream`, `beta.threads.runs.stream`, `beta.threads.runs.submit_tool_outputs`, `beta.threads.runs.submit_tool_outputs_and_poll`, `beta.threads.runs.submit_tool_outputs_stream`, `completions.create`, `decisions.create`, `evals.runs.create`, `fine_tuning.alpha.graders.run`, `fine_tuning.jobs.create`, `images.create_variation`, `images.edit`, `images.generate`, `live.connect`, `live.create`, `live.forks.connect`, `live.sessions.accept`, `live.sessions.fork`, `live.sideband.connect`, `realtime.calls.accept`, `realtime.calls.create`, `realtime.connect`, `responses.compact`, `responses.connect`, `videos.create`, `videos.create_and_poll`, `videos.create_character`, `videos.edit`, `videos.extend`, `videos.remix`
- **Anthropic** (`Anthropic` and `AsyncAnthropic` clients): `beta.deployments.run`, `beta.dreams.create`, `beta.environments.work.poller`, `beta.environments.work.worker`, `beta.messages.batches.create`, `beta.sessions.create`, `beta.sessions.events.send`, `beta.sessions.events.stream`, `beta.sessions.events.tool_runner`, `beta.sessions.threads.events.stream`, `messages.batches.create`
- **Google Gemini** (`google.genai` `Client` and `Client.aio`): `batches.create`, `batches.create_embeddings`, `caches.create`, `caches.update`, `file_search_stores.import_file`, `file_search_stores.upload_to_file_search_store`, `interactions.create`, `live.connect`, `live.music.connect`, `models.edit_image`, `models.recontext_image`, `models.segment_image`, `models.upscale_image`, `triggers.run`, `tunings.tune`
- **Google Vertex AI** (`vertexai` classes): `batch_prediction.BatchPredictionJob.submit`, `caching.CachedContent.create`, `caching.CachedContent.update`, `language_models.ChatModel.tune_model`, `language_models.ChatModel.tune_model_rlhf`, `language_models.ChatSession.send_message`, `language_models.ChatSession.send_message_async`, `language_models.ChatSession.send_message_streaming`, `language_models.ChatSession.send_message_streaming_async`, `language_models.CodeChatModel.tune_model`, `language_models.CodeChatSession.send_message`, `language_models.CodeChatSession.send_message_async`, `language_models.CodeChatSession.send_message_streaming`, `language_models.CodeChatSession.send_message_streaming_async`, `language_models.CodeGenerationModel.batch_predict`, `language_models.CodeGenerationModel.predict`, `language_models.CodeGenerationModel.predict_async`, `language_models.CodeGenerationModel.predict_streaming`, `language_models.CodeGenerationModel.predict_streaming_async`, `language_models.CodeGenerationModel.tune_model`, `language_models.TextEmbeddingModel.batch_predict`, `language_models.TextEmbeddingModel.deploy_tuned_model`, `language_models.TextEmbeddingModel.tune_model`, `language_models.TextGenerationModel.batch_predict`, `language_models.TextGenerationModel.predict`, `language_models.TextGenerationModel.predict_async`, `language_models.TextGenerationModel.predict_streaming`, `language_models.TextGenerationModel.predict_streaming_async`, `language_models.TextGenerationModel.tune_model`, `language_models.TextGenerationModel.tune_model_rlhf`, `preview.caching.CachedContent.create`, `preview.caching.CachedContent.update`, `preview.vision_models.ImageCaptioningModel.get_captions`, `preview.vision_models.ImageGenerationModel.upscale_image`, `preview.vision_models.ImageQnAModel.ask_question`, `preview.vision_models.ImageSegmentationModel.segment_image`, `preview.vision_models.ImageTextModel.ask_question`, `preview.vision_models.ImageTextModel.get_captions`, `preview.vision_models.MultiModalEmbeddingModel.get_embeddings`, `preview.vision_models.WatermarkVerificationModel.verify_image`, `tuning.sft.train`, `vision_models.ImageCaptioningModel.get_captions`, `vision_models.ImageGenerationModel.upscale_image`, `vision_models.ImageQnAModel.ask_question`, `vision_models.ImageTextModel.ask_question`, `vision_models.ImageTextModel.get_captions`, `vision_models.MultiModalEmbeddingModel.get_embeddings`
- **AWS Bedrock** (boto3 bedrock-runtime client; Claude models are metered when REVENIUM_BEDROCK_TRANSPORT is on): `apply_guardrail`, `converse`, `converse_stream`, `invoke_guardrail_checks`, `invoke_model`, `invoke_model_with_response_stream`, `start_async_invoke`
- **Perplexity** (`Perplexity` and `AsyncPerplexity` clients): `async_.chat.completions.create`, `browser.sessions.create`, `contextualized_embeddings.create`, `embeddings.create`, `responses.create`, `search.create`
- **LiteLLM** (the `litellm` module): `aadapter_completion`, `aadapter_generate_content`, `acode_interpreter_tool`, `acompact_responses`, `acreate_batch`, `acreate_fine_tuning_job`, `acreate_run`, `acreate_sandbox`, `adapter_completion`, `adecisions`, `agenerate_content`, `aimage_edit`, `aimage_generation`, `aimage_variation`, `aingest`, `anthropic_interface.messages.acreate`, `anthropic_interface.messages.create`, `anthropic_messages`, `anthropic_messages_handler`, `aocr`, `aquery`, `arealtime_calls`, `arerank`, `aresponses`, `aresponses_api_with_mcp`, `aresponses_with_retries`, `arun_code`, `arun_thread`, `arun_thread_stream`, `asearch`, `aspeech`, `asystemone`, `atranscription`, `avideo_create_character`, `avideo_edit`, `avideo_extension`, `avideo_generation`, `avideo_remix`, `compact_responses`, `config_completion`, `create_batch`, `create_fine_tuning_job`, `create_run`, `decisions`, `image_edit`, `image_generation`, `image_variation`, `ingest`, `nlp_cloud_chat_completion`, `ocr`, `query`, `replicate_chat_completion`, `rerank`, `responses`, `responses_with_retries`, `run_thread`, `run_thread_stream`, `search`, `speech`, `systemone`, `transcription`, `video_create_character`, `video_edit`, `video_extension`, `video_generation`, `video_remix`
- **Ollama** (`Client`, `AsyncClient` and the `ollama` module): `systemone`, `web_fetch`, `web_search`
- **fal.ai** (`SyncClient`, `AsyncClient` and the `fal_client` module): `realtime`, `realtime_async`, `ws_connect`, `ws_connect_async`
- **LangChain** (embeddings models from vendors other than OpenAI, Azure OpenAI and Ollama, such as `MistralAIEmbeddings`): `Embeddings.aembed_documents`, `Embeddings.aembed_query`, `Embeddings.embed_documents`, `Embeddings.embed_query`
<!-- not-metered:end -->

Raw boto3 calls to AWS Bedrock send no usage record for any model unless `REVENIUM_BEDROCK_TRANSPORT` is set to `1` or `true`; with it set, Claude models on `invoke_model`, `invoke_model_with_response_stream`, `converse` and `converse_stream` are metered, and other models still are not. Claude through `AnthropicBedrock` is metered either way. `boto3.client("bedrock-agent-runtime")` (`invoke_agent`, `retrieve_and_generate`), Bedrock batch inference through `boto3.client("bedrock").create_model_invocation_job`, and Vertex AI `evaluation`, `agent_engines` and `rag` are also not metered and are outside the check.

LiteLLM's `transcription` and `atranscription` calls to OpenAI models are recorded when `revenium_middleware.openai` is also imported, because LiteLLM sends them through the OpenAI SDK, which that integration meters; that record carries none of the `usage_metadata` passed to LiteLLM. Without that import they are not recorded.

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
print(status.evicted_count)   # undelivered events dropped because the buffer was full
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

Outcome, job and job-type economics calls go to the same Revenium address as usage records: `REVENIUM_METERING_BASE_URL`, or `https://api.revenium.ai` when it is unset. Set `REVENIUM_PROFITSTREAM_BASE_URL`, or pass `profitstream_base_url=`, only when outcomes live at a different address.

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
- **Auto-FAILED:** if an unhandled exception escapes the block, the context automatically reports `execution_status="FAILED"` (the exception's class in `metadata.error`, `metadata.errorType` and the outcome reason; its message too when `REVENIUM_CAPTURE_ERROR_MESSAGES=true`, redacted as for failed tool events and capped at 2048 characters; an exception from a provider's HTTP error, which can quote the request, sends its class only unless `REVENIUM_CAPTURE_PROMPTS=true` as well) and always re-raises the original exception. It skips the FAILED when a `report_outcome` or `amend_outcome` call succeeded, the platform answered that the job already has an outcome, the exception escaping the block is the one your own `report_outcome`/`amend_outcome` raised, or that call found no such job (404). So a report you caught that failed, followed by a crash, still gets FAILED. If the caught report was in fact stored (a timeout after the platform saved it), the automatic FAILED is refused as already reported and only logged; the stored outcome stands.
- **Job id format:** Revenium refuses any other job id on AI calls, so a job id must start with an ASCII letter or digit, contain only ASCII letters, digits, `.`, `_` and `-`, be at most 255 characters, and not be `types` or `conversion-funnel`. Entering `JobContext` with any other id raises `ValueError` before any call is made. `JobContext.attach()` still accepts such an id, for reporting on a job created with `create_job`. An id from `usage_metadata`, `@track_job`, `REVENIUM_AGENTIC_JOB_ID` or proxy headers that breaks the rule is left off the call: the call is still metered, a warning is logged once per id, and the call does not fall back to the surrounding job's id. AI calls never create a job under such an id, so `report_outcome` on it fails with a 404 after its retries unless the job was created with `create_job`.
- **Blocking:** outcome calls are synchronous HTTP requests with retries; tune how long they may block with the `retry_attempts`, `retry_initial_seconds`, and `retry_max_seconds` arguments, accepted by the `JobContext` constructor, `JobContext.attach()`, `get_outcome_history()`, the job reads in [Reading Jobs Back](#reading-jobs-back), and the CrewAI wrapper's `report_job_outcome`/`amend_job_outcome`.
- **Retry after a gateway error:** when a gateway answers 502, 503 or 504 and the retry is then refused as already reported, the SDK reads the job once and treats the report as delivered if the stored outcome is the one it sent, unamended. An identical outcome stored by another writer in that gap is reported as success too; the reporter label (`reported_by`) and `metrics` facts are not checked. A timeout or connection error on the report itself is not retried.
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

The handle also records that version, so the retry above works with or without passing `expected_entity_version=` explicitly. `current_entity_version` is `None` when the conflict body carries no version; read it with `get_job(job_id)["entityVersion"]`; a retry without it is unlocked (last-write-wins). `get_outcome_history()` rows carry an `amendment_sequence`, not an entity version. History tells you *what* changed, never which version to retry with.

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
- **One active fact per key:** a job keeps one active fact for each metric key. Appending a key the job already has supersedes the earlier fact (the earlier fact is deactivated, not deleted), and the platform refuses that append with a 400 unless the entry carries a `reason`. The same key twice in one request is refused. The SDK sends entries as written and never dedupes them. `metrics=` on `amend_outcome()` records facts as part of the amendment; a repeated key there also supersedes the earlier fact, and the amendment's own reason (or the platform's automated reason for an API-key caller) is used when an entry has none.
- **Not part of outcome history:** `get_outcome_history()` returns the outcome revisions only — appended facts do not appear in those rows.
- **Retries:** an append is retried only on `429`, which proves the platform rejected the request before recording anything. A `502`/`503`/`504` is raised instead of retried: the facts may already be recorded, and resending a key that is already recorded is a correction, refused without a `reason` and, with one, superseding the fact, so the decision to resend is yours (check the recorded facts first).
- **Locking is unaffected:** appending facts does not change the job's `entityVersion`, so the handle keeps the version it recorded and a following `amend_outcome()` still locks against it. (`report_outcome()` and `amend_outcome()` clear the recorded version when their response carries none, because those calls advance it; an append does not.)
- **Why it matters:** AI Alerts evaluate `QUALITY_RATE` from these facts, so a job whose integration emits none is invisible to those rules.

### Reading Jobs Back

These calls need a write-scope key (`rev_sk_`), like every other outcome call. A metering key (`rev_mk_`) is rejected before any request.

```python
from revenium_middleware import (
    get_conversion_funnel, get_job, get_job_roi, get_job_transactions,
    get_job_types, list_jobs,
)

page = list_jobs(job_type="loan_processing", execution_status="SUCCESS", size=50)
for job in page["content"]:
    print(job["agenticJobId"], job["outcomeType"])

version = get_job("loan-app-12345")["entityVersion"]
types = get_job_types()
roi = get_job_roi("loan-app-12345")["roi"]
calls = get_job_transactions("loan-app-12345")["transactions"]
rate = get_conversion_funnel(
    start_date="2026-09-01T00:00:00Z", end_date="2026-09-30T23:59:59Z",
)["conversionRate"]
```

- **Results are the API's JSON:** each call returns dicts with the field names the API sends (camelCase), so a field the platform adds later reaches you without an SDK upgrade.
- **Paging:** `list_jobs` returns `{"content": [...], "page": {...}}`; `page` is 0-based and only the filters you pass are sent.
- **Missing job:** a job that does not exist raises `httpx.HTTPStatusError` (404).
- **Team and retries:** team resolution and the retry knobs work as for `get_outcome_history()`.

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

In an `emit_completion` payload the agent is named by `agent`. `agentName` and `currency` are not sent (Revenium does not store them); `emit_completion` names them in its one-time warning about unsent keys.

`emit_completion` and `emit_tool_event` queue the record and return at once, like every automatic record: the SDK's background threads send it with the same timeouts and retries (see [Metering Delivery and Timeouts](#metering-delivery-and-timeouts)), and a record that fails with a timeout, a refused connection or a 5xx goes to the store-and-forward buffer and is sent when Revenium is reachable again. Delivery errors are logged, not raised; `emit_completion` still raises `ValueError` before anything is queued when the payload is missing a required key. Both calls return the payload you passed.

Records go to the address set in code with `initialize_metering(base_url=...)` or a `metering_key(..., base_url=...)` block when one is set, else `REVENIUM_METERING_BASE_URL`; pass `meter_base_url=` to `AgenticOutcomeSettings` to pin another address. They are always sent under the settings' `api_key`, never a `metering_key()` key. `http_client=` serves jobs and outcomes only.

A script that emits many records and exits right away has `REVENIUM_SHUTDOWN_TIMEOUT_SECONDS` (default 5) to send the records still queued; raise it if records are reported unsent at exit.

`emit_tool_event` sends one `Idempotency-Key` per call and reuses it when the buffer replays the event, so a tool event replayed after a gateway error is stored once.

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

### Several calls in one block

- Each model call in a block gets its own key: `order-42`, then `order-42:2`, `order-42:3` and so on, in the order its usage record is delivered. That can differ from the order the calls were made, because records are delivered in the background. Every call in the block is recorded.
- A retry the SDK makes itself (an HTTP retry, or a replay from the delivery buffer) reuses the same key and is stored once.
- Rerunning the same block is refused call by call, up to the number of calls the original block made: each refusal is logged and counted as a metering error, and the rerun's provider spend is not recorded. Calls beyond the original count are recorded.
- A block key that ends in `:<number>`, such as `order-42:2`, can collide with another block's derived keys. Avoid that shape.
- Tool events are not keyed by the block; each tool event carries its own key (see [Tool Metering](#tool-metering)).
- fal queue jobs are not keyed by the block either; each job's record uses its own key, `fal-<request id>` (see [fal.ai](#falai)).
- If a derived key would be longer than 255 characters, the SDK uses the SHA-256 hex digest of the block key in its place, so the key stays valid and the same on every rerun.

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

`Idempotency-Key` must be 1–255 printable ASCII characters, with no leading or trailing spaces. `idempotency_key()` and `set_idempotency_key()` raise `ValueError` for an empty key and for any other key that breaks this rule (an accented or non-Latin letter, a line break, a tab, a leading or trailing space, or more than 255 characters), before anything in the block runs. UUID v4 is the recommended format and what the SDK generates by default.

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

Supports chat completions, streaming, embeddings, function calling, vision/multimodal, and audio (speech, transcription and translation).

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

A chat or Responses call the provider refuses (a 400, 429 or 500, for example), streamed or not, sends one record with stop reason `ERROR`, zero tokens and the error's class and HTTP status as its error reason (plus the provider's redacted error message when both `REVENIUM_CAPTURE_ERROR_MESSAGES` and `REVENIUM_CAPTURE_PROMPTS` are `true`; a provider's error can quote the request, so it is never sent with prompt capture off), cut to 255 characters with its "(HTTP <status>)" ending kept; a stream that breaks off part-way is recorded with stop reason `ERROR` too. The HTTP status is also sent as the record's error code; a call that got no HTTP error answer (a connection error, a timeout, or a stream that broke off) has none. Your code still gets the original exception. Embeddings, `parse()` and calls a spending cap refuses send no error record.

#### Audio

`audio.transcriptions.create`, `audio.translations.create` and `audio.speech.create` (sync, async, `with_raw_response` and `with_streaming_response`, on `OpenAI` and `AzureOpenAI`) each send one audio record, with `usage_metadata` attribution like a chat call. Spending caps apply to them as they do to chat.

```python
transcript = client.audio.transcriptions.create(model="whisper-1", file=open("call.mp3", "rb"))
speech = client.audio.speech.create(model="tts-1", voice="alloy", input="Your order has shipped.")
```

What each record carries:
- **Transcription and translation:** the audio seconds OpenAI bills (`usage.seconds`, else the `duration` of a `verbose_json` response), or, for models OpenAI bills by token (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`), the input, output and audio input tokens. A streamed transcription is recorded once, when the stream ends or is closed.
- **Speech:** the number of characters in `input`, plus `voice`, `speed` and `response_format`. The audio itself is never read.

Some calls report no duration, so their record is stored with a duration of zero and no cost: transcriptions with `response_format` `text`, `srt` or `vtt`, translations in the default `json` format (use `verbose_json` to get the duration), anything called through `with_streaming_response`, and a `with_raw_response` call with `stream=True`. Revenium may also show $0 for models it prices by token, such as `gpt-4o-mini-tts`, as it does for the Go and Node SDKs.

If a LiteLLM gateway with `ReveniumGuardrail` also imports `revenium_middleware.openai`, an OpenAI audio call proxied through it is recorded twice; import only the LiteLLM integration in the gateway process.

#### Azure OpenAI

The middleware automatically detects Azure OpenAI when using `AzureOpenAI()` and resolves deployment names to standard model names for accurate pricing. The model Azure reports (for example `gpt-4.1-nano-2025-04-14`) is recorded as is; a custom deployment name is mapped to its model family.

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

#### xAI (Grok)

Point the OpenAI client at xAI and calls are recorded with provider `XAI`, so xAI spend is priced and reported as xAI rather than OpenAI.

A spending cap on a Grok model also stops calls that use the short names xAI publishes for it, such as `grok-4.20`; see [Cost Controls / Enforcement](#cost-controls--enforcement) for how model filters treat vendor aliases.

On a Grok reasoning model the reasoning tokens xAI bills are counted as output, and each record carries xAI's own charge for the call (`cost_in_usd_ticks`) as its total cost. This applies when the client's base URL is on `x.ai`; Grok reached through a proxy or gateway is recorded as the OpenAI-compatible call it appears to be.

```python
import os
from openai import OpenAI
import revenium_middleware.openai

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

response = client.chat.completions.create(
    model="grok-4.20-0309-non-reasoning",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

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

**Note:** The middleware meters `messages.create` (including `create(stream=True)`), `messages.stream` and `messages.parse`, and the same calls on `client.beta.messages` (`create`, `stream`, `parse` and each step of `tool_runner`), sync and async, including their `with_raw_response` and `with_streaming_response` forms, on the Anthropic, Bedrock and Vertex AI clients. Calls listed under [Calls the SDK does not meter](#calls-the-sdk-does-not-meter), message batches among them, are not metered.

A call the provider refuses sends one record with stop reason `ERROR`, zero tokens and the error's class and HTTP status as its error reason (plus the provider's redacted error message when both `REVENIUM_CAPTURE_ERROR_MESSAGES` and `REVENIUM_CAPTURE_PROMPTS` are `true`; a provider's error can quote the request, so it is never sent with prompt capture off; at most 255 characters, keeping its "(HTTP <status>)" ending), as does a `create(stream=True)` stream that breaks off part-way. The HTTP status is also sent as the record's error code; a call that got no HTTP error answer (a connection error, a timeout, or a stream that broke off) has none. Your code still gets the original exception. Calls served through the Bedrock fast path, and calls a spending cap refuses, send no error record.

#### AWS Bedrock

The middleware meters Claude on AWS Bedrock through `AnthropicBedrock`, with automatic detection.

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
| `REVENIUM_BEDROCK_DISABLE` | Set to `1` to disable Bedrock support for Anthropic clients (Bedrock detection only - Foundry detection is unaffected). Raw boto3 calls metered with `REVENIUM_BEDROCK_TRANSPORT`, and LangChain's `ChatBedrock` and `ChatBedrockConverse`, are still recorded as AWS | Not set |
| `REVENIUM_BEDROCK_TRANSPORT` | Set to `1` or `true` to meter Claude models called through a raw `boto3.client("bedrock-runtime")` (`invoke_model`, `invoke_model_with_response_stream`, `converse`, `converse_stream`); other Bedrock models are never metered this way | Not set |

**AWS authentication**: an `AnthropicBedrock` client uses its own `aws_access_key` / `aws_secret_key` / `aws_session_token`, `aws_profile` or Bedrock `api_key`, exactly as without the middleware. A plain `anthropic.Anthropic` client with a Bedrock `base_url` uses the standard credential chain: environment variables, `~/.aws/credentials`, IAM roles, AWS SSO. Required permissions: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`.

**Raw boto3 calls, application inference profiles and Provisioned Throughput**: with `REVENIUM_BEDROCK_TRANSPORT=1`, raw `boto3.client("bedrock-runtime")` calls to Claude (`invoke_model`, `invoke_model_with_response_stream`, `converse`, `converse_stream`) are metered and capped. A call written as a cross-Region inference profile ARN or a foundation model ARN is recorded and priced under the model id the ARN names (`us.anthropic.claude-haiku-4-5-20251001-v1:0`, `anthropic.claude-haiku-4-5-20251001-v1:0`), as the same call by that plain id is. A Claude call through an application inference profile ARN (`arn:aws:bedrock:<region>:<account>:application-inference-profile/<id>`) also needs `bedrock:GetInferenceProfile`, and one through a Provisioned Throughput ARN (`arn:aws:bedrock:<region>:<account>:provisioned-model/<id>`) needs `bedrock:GetProvisionedModelThroughput`, because neither ARN names the model: the SDK looks each ARN up once per process, in the background, starting at its first call, and from then on records and caps the call under the Claude model it runs. The lookup never delays your call: calls made before it answers, usually only the first in a process, are neither metered nor capped. The lookup uses the proxy, client certificate and CA bundle set on your boto3 client. A Provisioned Throughput call is recorded under its foundation model (for example `anthropic.claude-haiku-4-5-20251001-v1:0`), so the cost Revenium shows is the on-demand equivalent of its tokens, not the hourly price AWS bills, and a cost cap fills at that rate: a breached cost BLOCK cap can stop calls on capacity you have already paid for. Without the permission the call still goes through, unmetered and uncapped, and the SDK logs one warning per ARN. A lookup that fails for a temporary reason (an outage, a dropped connection, an expired session token) is retried every 5 minutes in the background. Profiles and Provisioned Throughputs that run a non-Claude model are not metered. A profile copied from a US or EU cross-Region profile is recorded and priced as that profile (for example `us.anthropic.claude-haiku-4-5-20251001-v1:0`), so its cost matches the regional rate AWS bills. A profile copied from an Australia, Japan, APAC, India or global cross-Region profile, or on a single Region, is recorded under the model id (`anthropic.claude-haiku-4-5-20251001-v1:0`). A cap on either id or on the Claude model name (`claude-haiku-4-5-20251001`) applies to the call. With cost controls on, a sync `AnthropicBedrock` call by an application inference profile ARN is also capped under the Claude model the profile routes to, using the same once-per-process lookup and the same `bedrock:GetInferenceProfile` permission; without the permission it is still metered but cannot be capped by model. A failed lookup is remembered only for the credentials that made it, so a client with the permission still looks the profile up and is metered and capped. An `AsyncAnthropicBedrock` call by a profile is capped that way once the profile has been looked up by a sync or raw boto3 call in the same process.

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

Supports chat completions, streaming, embeddings, image generation (Imagen), and vision/multimodal. The Google AI SDK (`google-genai`) is the recommended path for both the Gemini Developer API (API key) and Vertex AI (`genai.Client(vertexai=True, ...)`): it counts chat tokens on both, and embedding tokens on Vertex AI. The SDK still meters calls made through the Vertex AI SDK (`vertexai`), but Google deprecated that SDK's generative AI modules (`generative_models`, `language_models`, `vision_models`) and its releases after June 24, 2026 do not include them; see [Google's notice](https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk). Use google-genai with `vertexai=True` for new work.

```bash
# Google AI SDK (Gemini Developer API, or Vertex AI with vertexai=True)
pip install "revenium-python-sdk[google-genai]"

# Vertex AI SDK (vertexai package)
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

The same call on Vertex AI, with the same metering:

```python
client = genai.Client(vertexai=True, project="your-gcp-project", location="us-central1")
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Hello! Introduce yourself in one sentence.",
    usage_metadata={"organizationName": "AcmeCorp", "task_type": "chat"}
)
print(response.text)
```

#### Vertex AI SDK (deprecated by Google)

Google deprecated the generative AI modules of this SDK; see the notice above. Use the google-genai example above for new code.

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
| Production applications | Google AI SDK with `vertexai=True` | Full token counting, including embeddings, on Vertex AI |
| Embeddings-heavy workloads | Google AI SDK with `vertexai=True` | Vertex AI returns a token count for every embedded text |
| Existing Vertex AI SDK code | Vertex AI SDK (deprecated by Google) | Still metered; migrate to google-genai with `vertexai=True` |

**Note:** On Vertex AI (`genai.Client(vertexai=True, ...)`) google-genai embeddings are recorded with the input tokens Google returns for every text. The Gemini API (API key) returns no token count for embeddings, so those calls are recorded with 0 tokens, and still tracked.

A `generate_content` or `generate_content_stream` call through the Google AI SDK (the Gemini API, or Vertex AI with `genai.Client(vertexai=True)`) that Google refuses (a 429 or 500, for example), sync, async or streamed, sends one record with stop reason `ERROR`, zero tokens and the error's class and HTTP status as its error reason; a stream that breaks off part-way is recorded with stop reason `ERROR` too. The HTTP status is also sent as the record's error code; a call that got no HTTP error answer (a connection error, a timeout, or a stream that broke off) has none. Your code still gets Google's own exception. Embeddings, Imagen, Veo, calls made with the `vertexai` library and calls a spending cap refuses send no error record.

**Images from Gemini image models:** an image made with `generate_content` or `generate_content_stream` on a Gemini image model (for example `gemini-3.1-flash-lite-image`), through google-genai (the Gemini API, or `genai.Client(vertexai=True)`) or the `vertexai` library, sync, async or streamed, sends two records that share the call's trace, job and customer. An image record (operation type `IMAGE`, transaction id the chat record's plus `-image`) carries the number of images and is priced at the model's per-image price in Revenium; the chat record carries only the answer's text tokens, with the image tokens Google reports removed. Known limits: the per-image price is one fixed price per model, so 2K and 4K images are under-billed; interim "thought" images are not counted, only the final images are; when Google does not report the image's token share, the image tokens also stay on the chat record; a call that fails or a stream that breaks off is recorded as chat only. Gemini images made through LiteLLM (`gemini/...`), the LangChain unified handler fallback or griptape are still recorded as chat.

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

**Supported endpoints:** `chat()`, `generate()`, `embed()` and the legacy `embeddings()`, whether called at module level (`ollama.chat()`) or on an explicit client such as `ollama.Client(host="http://my-ollama:11434")` or `ollama.AsyncClient(...)`, streamed or not.

A `chat()` or `generate()` call the Ollama server refuses (a 500, for example), streamed or not, sends one record with stop reason `ERROR`, zero tokens and the error's class and HTTP status as its error reason; a stream that breaks off part-way is recorded with stop reason `ERROR` too. The HTTP status is also sent as the record's error code; a call that got no HTTP error answer (a connection error, a timeout, or a stream that broke off) has none. Your code still gets Ollama's own exception. Embeddings and calls a spending cap refuses send no error record.

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

Client mode meters `completion`, `acompletion`, `text_completion`, `atext_completion`, `completion_with_retries` and `acompletion_with_retries`, streamed or not, and `embedding` / `aembedding` as embedding calls. When you stream without `stream_options={"include_usage": True}`, the middleware asks LiteLLM for the usage chunk itself and keeps it out of your stream. LiteLLM adds that chunk for every provider, including ones it does not send `stream_options` to. `litellm.responses` and `litellm.aresponses` are not metered, except on providers LiteLLM serves through `completion`. A completion that fails, or a stream that breaks off, sends one record with stop reason `ERROR`, zero tokens for a call that never started, and the error's class and HTTP status as its error reason (plus the provider's redacted error message when both `REVENIUM_CAPTURE_ERROR_MESSAGES` and `REVENIUM_CAPTURE_PROMPTS` are `true`; a provider's error can quote the request, so it is never sent with prompt capture off); your code still gets the original exception. The HTTP status is also sent as the record's error code; a call that got no HTTP error answer (a connection error, a timeout, or a stream that broke off) has none. LiteLLM gives a connection error a 500 and a timeout a 408 that no server answered, so those get no error code and no HTTP status in their error reason, including the `InternalServerError` LiteLLM raises for a refused connection on its `openai/` and `anthropic/` providers. LiteLLM also raises a few real provider answers, such as a Vertex AI 502, as connection errors, and those get no error code either. Failure records sent by the LiteLLM gateway (proxy) do not carry the error code yet.

Inside a LiteLLM proxy running `ReveniumGuardrail` with `default_on: true` and `post_call` among its modes, client mode leaves each proxied request to the guardrail, so the request is recorded once, with the guardrail's subscriber. The row names the deployment's model (the `litellm_params.model` behind the `model_name` your callers use, without its provider prefix), so a call to an alias is priced as the model it reached. The guardrail also meters streamed `/v1/responses` calls (Codex CLI, the OpenAI Agents SDK) for every provider. Client mode still records the proxied calls the guardrail does not meter, which today are streamed `/v1/completions` calls, and every call your own code in the proxy's process makes to LiteLLM directly.

#### Proxy Mode

`ReveniumGuardrail` is the LiteLLM proxy integration. It is a LiteLLM
`CustomGuardrail` that **enforces the caller's budget before** the proxied call and
**meters usage after** it: successes, failures and streamed responses alike,
except calls a spending cap refuses.

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
guardrail to that one event type. `pre_call` alone enforces without metering.
`post_call` alone meters successful calls without enforcing, but records no failed
calls: a failed call is recorded only for a request the Revenium guardrail's
`pre_call` hook ran on, so a request LiteLLM refuses for a wrong or missing key
sends no record. Adding `pre_call` enforces no budget unless
`REVENIUM_CIRCUIT_BREAKER_ENABLED=true`.

**Budget enforcement** reuses the SDK's own circuit breaker, so a proxy enforces
exactly what every other Revenium integration enforces — including department
budgets. An organization cap is checked against the organization the call names,
which a caller can choose by header unless you pin it to the key's team (see
"Organization caps and the organization header" below). It is opt-in via `REVENIUM_CIRCUIT_BREAKER_ENABLED=true`; see
[Cost Controls](#cost-controls). A blocked call never reaches the provider, the
caller receives HTTP 429, and no failed-call record is sent, as with the OpenAI and
Anthropic integrations:

```json
{"error": {"message": "Request blocked by Revenium enforcement rule: Team Budget",
           "type": "budget_exceeded", "guardrail": "revenium", "model": "gpt-4o",
           "budgets": [{"name": "Team Budget", "ruleId": 7, "threshold": 10.0,
                        "currentValue": 11.5, "resetsAt": "2026-10-01T00:00:00Z"}]}}
```

When the rule carries a deny message, the body's `message` ends with it, as `BudgetExceededError.message` does (see [Exception Contract](#exception-contract)).

Enforcement **fails open**: if the enforcement path is unreachable or misbehaves,
the call proceeds, and it proceeds without waiting. The pre-call check reads the
rules the SDK already has cached and refreshes them on a background thread, so a
Revenium endpoint that hangs, refuses connections, returns errors or rejects your
key (403) neither blocks nor slows the proxy. A failed refresh is retried once
per `REVENIUM_CB_POLL_INTERVAL_SECONDS`, not once per request. Right after the
proxy starts, before the first rules fetch completes, no rules are cached and
calls go through; see [Fail-Open vs Fail-Closed](#fail-open-vs-fail-closed).
Metering is likewise non-disruptive — nothing in the post-call path can turn a
successful LLM call into an error for the client.

**Attribution** travels as `x-revenium-*` request headers (subscriber, organization,
product, trace, task type, agent, subscription, quality score, and `x-revenium-effort`
for reasoning effort), with the calling virtual key's metadata as the fallback —
`revenium_user_id`, `revenium_organization_name`, `revenium_key_name`, and
`revenium_agentic_job_*`. Agentic job tags (`x-revenium-agentic-job-id`, `-name`,
`-type`, `-version`) ride along for cost/ROI correlation; the job id is required for
the others to be recorded. A malformed header value, such as a quality score of
`high` or an effort of `very high`, is left off the record with a WARNING in the
gateway log; the call is still recorded.

**Trace id.** The guardrail takes a record's trace id from `x-revenium-trace-id`.
Without that header it takes the LiteLLM session id the caller sent: the
`x-litellm-session-id` or `x-litellm-trace-id` header, or `litellm_session_id` or
`metadata.session_id` in the request body (`litellm_metadata.session_id` on
`/v1/messages`). Calls a customer groups into one LiteLLM session therefore read
as one trace in Revenium. Failing both, it uses Claude Code's
`x-claude-code-session-id`, so one Claude Code session is one trace. A call with
none of these has no trace id. The id LiteLLM generates for every request is never
used, because it would make each call a trace of its own, and neither is a session
id LiteLLM minted itself under its `missing_session_id: generate` policy. The
deprecated callback reads only `x-revenium-trace-id` and `x-claude-code-session-id`.

**Organization caps and the organization header.** By default the
`x-revenium-organization-name` header (and the retired `x-revenium-organization-id`)
wins over the calling virtual key's team alias, for the budget check and the record
alike. That means anyone holding a team's key can send another organization's name
and move their spend off their own organization's cap: the call is checked against,
and recorded under, the name they sent. To make the key's team decide instead, set

```bash
export REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM=true
```

With it on, a call made with a key that belongs to a LiteLLM team is checked and
recorded under that team's alias whatever organization header it sends, so the team
alias is the organization: renaming the team in LiteLLM renames where its spend is
booked. A key with no team alias (none set, or blank) still takes the organization
from the header; on the guardrail, a key with neither falls back to its
`revenium_organization_name` metadata. `1`, `yes` and `on` also turn the setting on.
Unset, `false`, `0`, `no` and `off` leave it off; any other value leaves it off too,
and the proxy logs a warning at startup naming the setting. The deprecated callback
honours the same setting for its record.

**Trace fields** come from the gateway host's own settings, and a matching header wins
per call: `REVENIUM_ENVIRONMENT` (or `ENVIRONMENT`, or `DEPLOYMENT_ENV`) and
`x-revenium-environment`; `REVENIUM_REGION` (or `AWS_REGION`, `AWS_DEFAULT_REGION`,
`AZURE_REGION`, `GCP_REGION`, `GOOGLE_CLOUD_REGION`) and `x-revenium-region`;
`REVENIUM_CREDENTIAL_ALIAS` and `x-revenium-credential-alias`;
`REVENIUM_TICKET_ID` and `x-revenium-ticket-id`; `REVENIUM_TRANSACTION_NAME`, `REVENIUM_TRACE_TYPE`, `REVENIUM_TRACE_NAME`,
`REVENIUM_PARENT_TRANSACTION_ID`, `REVENIUM_RETRY_NUMBER` and `REVENIUM_SKILL_*`, with
the headers `x-revenium-transaction-name`, `-trace-type`, `-trace-name`,
`-parent-transaction-id`, `-retry-number`, `-skill-name`, `-skill-source`,
`-skill-kind`, `-skill-plugin-name`, `-skill-marketplace-name` and
`-skill-invocation-trigger`. The gateway does not fall back to the task type for the
transaction name, and sends a retry number only when the header or
`REVENIUM_RETRY_NUMBER` sets one. The credential alias names the credential only on
calls with no subscriber id or email and no virtual key alias; most virtual keys carry
their owner's email, so those calls keep the owner's own credential. A spending cap on
a credential alias therefore covers only calls with no subscriber identity, and on
those calls a caller's `x-revenium-credential-alias` wins over the host's alias, for
the cap check and the record alike. The agent's own version rides in
`x-revenium-agent-version`, per call only with no host setting, and is cut to 64
characters.

**Who made the call.** By default a proxied call is metered and budget-checked as the
owner of the virtual key it used, which is right when every person has their own key.
When several people share one key, have each client send
`x-revenium-subscriber-email` (and optionally `x-revenium-subscriber-id`) naming the
person making the call. Per-person and department budgets are looked up by that email,
so without it a shared key's calls all count against its owner, and a blocked
developer keeps getting through on the owner's budget. A call that sends someone
else's email without an id carries no id at all, rather than the key's
`revenium_user_id`, so it is checked against that person's balance and not the owner's. The captured header wins over a
copy of the same key in the request's `metadata`; a blank value is ignored. Both are
taken as the caller sends them: a shared key trusts its users to name themselves. On
a team with strict ingestion enabled, the email must belong to a registered user, or
the call's metering row is rejected instead of falling back to the key owner.

```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: sk-shared-team-key" \
  -H "x-revenium-subscriber-email: dev@acme.com" \
  ...
```

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

**It is on by default.** To opt out, with no `request-id` header added (each
`/v1/messages` record then carries a per-request id the proxy assigns, never
Anthropic's `msg_` id):

```bash
export REVENIUM_LITELLM_SHARED_CALL_ID=false
```

Blank or unset means on; `true`, `1`, `yes`, `on` mean on; any other value turns it off and the guardrail logs that at startup.

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

While the callback is listed, client mode leaves each proxied completion to it, so the
call is recorded once, with the subscriber, organization and product from the
`x-revenium-*` headers. Proxied embeddings are recorded by client mode, without those
headers, as the callback has never metered them.

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
`revenium_middleware.litellm.client` as documented above. The LiteLLM client
wrapper runs the same pre-call circuit breaker with the caller's `usage_metadata`
(and any `@revenium_metadata` identity) and raises `BudgetExceededError` before
LiteLLM is called, so LiteLLM's retries never send a blocked call. That check runs
only outside the proxy: requests the proxy serves are enforced by this guardrail.
Both the guardrail and the client wrapper
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
pip install "revenium-python-sdk[openai]" crewai
```

Pre-built wrapper for tracking CrewAI agent executions. Use the extra for your agents' models: `[openai]`, `[anthropic]`, `[google-genai]`, or `[litellm]` for models CrewAI sends through LiteLLM. CrewAI 1.x supports Python 3.10 to 3.13.

**Import the middleware for your agents' models before you build the crew.** CrewAI 1.x calls OpenAI, Anthropic and Gemini through their own SDKs, not through LiteLLM, so those calls are recorded by the matching Revenium middleware:

```python
import revenium_middleware.openai            # OpenAI models, e.g. LLM(model="gpt-4o-mini")
import revenium_middleware.anthropic         # Anthropic models, e.g. LLM(model="anthropic/claude-3-5-haiku-latest")
import revenium_middleware.google.google_ai  # Gemini models, e.g. LLM(model="gemini/gemini-2.5-flash")
import revenium_middleware.litellm.client.middleware  # LLM(..., is_litellm=True) and models CrewAI routes to LiteLLM
```

Each model call the crew makes during `kickoff()` or `kickoff_async()`, including a crew built with `stream=True` as you iterate it, is recorded once and carries the wrapper's organization, subscription, product, trace and agentic job fields. Importing more than one of these is safe: a call is still recorded once. When an agent's model has no matching middleware imported, the wrapper logs one warning naming the import to add.

What the wrapper does not cover:
- CrewAI's native Azure (`azure-ai-inference`), Bedrock (boto3 Converse) and Snowflake providers are not metered. Build those LLMs with `LLM(model=..., is_litellm=True)` and import the LiteLLM middleware.
- Only `kickoff()` and `kickoff_async()` are attributed. Calls started through the underlying crew's `akickoff()` or `kickoff_for_each*()` are not.

**Job outcome tracking:** pass the `agentic_job_*` kwargs to tie every LLM call in the crew to one agentic job, then report (or later amend) the job's business outcome. Requires a write-scope key (`rev_sk_`) — see [Agentic Outcomes](#agentic-outcomes-outcome-based-metering).

```python
from revenium_middleware.litellm.client.integrations.crewai import ReveniumCrewWrapper

crew = ReveniumCrewWrapper(
    agents=[support_agent],
    tasks=[triage_task],
    organization_name="AcmeCorp",
    subscription_id="82764738",
    product_name="Platinum",
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
    messages=[{"role": "user", "content": "Hello!"}],
    usage_metadata={"organizationName": "AcmeCorp"}
)
```

Both approaches work identically - the middleware automatically detects which SDK you're using. Both clients accept Revenium metadata as `usage_metadata=` or as `extra_body={"usage_metadata": ...}`, and the middleware removes it before the request reaches Perplexity. The one exception is the OpenAI client's `responses.stream()` helper, sync or async: it takes `usage_metadata=` only when `revenium_middleware.openai` is imported, and otherwise raises `TypeError` before any request is sent. With only `revenium_middleware.perplexity` imported, pass `extra_body={"usage_metadata": ...}` to that helper instead. Search, embeddings, contextualized embeddings, async chat and browser session calls on the native client are not metered or capped (see [Calls the SDK does not meter](#calls-the-sdk-does-not-meter)); metadata passed to them is removed and not used. An OpenAI-client embeddings call to Perplexity is recorded only when `revenium_middleware.openai` is imported.

A native Perplexity SDK `chat.completions.create` call that Perplexity refuses (a 400, 429 or 500, for example), streamed or not, sends one record with stop reason `ERROR`, zero tokens and the error's class and HTTP status as its error reason. The HTTP status is also sent as the record's error code; a call that got no HTTP error answer has none. Your code still gets Perplexity's own exception. A native stream that breaks off after it started, and calls a spending cap refuses, send no error record.

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

**Through LangChain:** `ChatPerplexity` keeps only token counts from Perplexity's answer, so the charge Perplexity reports (`usage.cost`, which includes its flat per-request fee) would never reach a LangChain record. When `perplexityai` is installed, importing `revenium_middleware.openai.langchain` therefore adds a read-only hook to the Perplexity SDK's `chat.completions.create` and `responses.create` (sync, async and streamed). Inside a LangChain run it hands the reported cost to the Revenium callback's record; it sends no record of its own, checks no spending cap, and leaves calls made outside a LangChain run untouched. Importing `revenium_middleware.perplexity` as well still works as above: its own record carries the cost for chat calls, and calls with a built-in tool (which go to the Responses API) are recorded by the callback.

**Examples:** `examples/perplexity/` - `getting_started.py`, `basic.py`, `streaming.py`, `example_decorator.py`

---

### fal.ai

Supports image, video, and audio generation through fal.ai with automatic media type detection. Language model endpoints (`fal-ai/any-llm`, the OpenRouter router `openrouter/router`) are recorded as chat completions with the tokens fal reports.

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

**Supported methods:** `run`, `subscribe`, `stream` and the queue flow (`submit`, then `status` and `result` or the handle's `get()`), on the module-level functions (`fal_client.run`, `fal_client.run_async`, and so on) and on clients you construct yourself (`fal_client.SyncClient(key=...)`, `fal_client.AsyncClient(key=...)`). Pass `usage_metadata` to `run`, `subscribe`, `stream`, `submit`, `result` or the handle's `get()`.

**Queued jobs** are metered once, when the result is fetched: `submit` and `status` send nothing, fetching the same result again sends nothing more, and a job whose result is never fetched is not metered. A stream is metered from its last event once you have read it to the end.

**Jobs fetched in another process:** the tags given to `submit` stay in the process that submitted the job, because fal has no field that carries them with it. When another process fetches the result (another worker, a webhook handler, the same service after a restart), pass the tags there: `fal_client.result(application, request_id, usage_metadata={...})` or `handle.get(usage_metadata={...})`. Without them the job is recorded with no organization, product, subscriber, trace or agent, and the SDK logs a warning naming the request id. In the process that submitted the job, the tags given at submit win, and tags given at fetch fill only the fields the submit left unset.

Every record of a queued job is sent under `fal-<request id>`, as both its transaction id and its Idempotency-Key, so fetching the same job again in another process does not bill it twice: Revenium refuses the repeat, and the SDK logs the refusal and counts it as one metering error. Revenium keeps the first record it receives for a transaction id, so this holds however long after the first fetch the repeat comes, and a later fetch without tags never replaces a tagged record.

**Media type detection:** The middleware automatically detects the type of media being generated (image, video, audio) based on the application name for accurate cost tracking.

**Environment variables:**
- `FAL_KEY` - Your fal.ai API key

---

### LangChain

Callback handler that automatically tracks LLM calls, chains, tools, and agent actions.

```bash
pip install "revenium-python-sdk[langchain]"
```

Wrap any LangChain chat model or LLM with `wrap()`; the Revenium callback handler is attached for you:

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

**Supported providers:** OpenAI, Anthropic, Google, AWS Bedrock, Azure OpenAI, Cohere, HuggingFace, Mistral, Ollama. Provider is auto-detected from LangChain class name or model name prefix.

**Perplexity cost:** a `ChatPerplexity` record carries the cost Perplexity reported, per-request fee included, for chat calls and for calls with a built-in tool such as web search. Importing this module installs a read-only hook on the Perplexity SDK for this (see [Perplexity](#perplexity)); the hook sends nothing itself.

**Text-completion models:** streamed calls from `langchain_openai.OpenAI` and `AzureOpenAI` (`stream`, `astream`, `invoke` with `streaming=True`) are recorded with the provider's token counts. Importing this module installs a read-only hook on the OpenAI SDK's legacy `completions.create` that asks the endpoint for its usage chunk during a LangChain stream and keeps that chunk out of your stream, except that a server whose usage chunk also carries an empty choice, such as the LiteLLM proxy, passes it through as one extra empty text chunk. The hook sends nothing itself, and a direct `completions.create` call outside LangChain is untouched and still not metered. A server that ignores the request gets a record with zero tokens and a warning. These streamed calls still send no record: `ainvoke` on a `streaming=True` model, a stream you stop reading early, and Azure OpenAI api-versions before `2024-09-01-preview`, which reject the usage request, so it is not sent there.

**Claude advisor and compaction passes:** a `ChatAnthropic` call metered by the callback alone sends one record per advisor or compaction pass when LangChain does not stream the call. Whenever LangChain streams it, only the main call is recorded, because LangChain keeps no per-pass usage on a streamed result: `stream`, `astream`, `streaming=True`, a `stream=True` call keyword, `astream_events`, or a LangGraph run streamed in `messages` mode, even when your code calls `invoke`. To record the passes for those calls too, also `import revenium_middleware.anthropic`, which covers every form.

**Attaching to an existing LLM:** use `attach_to()` to add tracking in-place, with any of the standard metadata fields (see [Metadata Fields](#metadata-fields)):

```python
from revenium_middleware.openai.langchain import attach_to

attach_to(llm, usage_metadata={
    "organizationName": "my_org",
    "productName": "my_product",
})
```

Calling `attach_to()` again on the same model replaces its Revenium handler rather than adding a second one. Attach once at startup with the fields every call shares. Do not call `attach_to()` per request on a model your requests share: overlapping requests would all be recorded under whichever person was attached last. To name the person per request, use one of these:

```python
from revenium_middleware import revenium_metadata
from revenium_middleware.openai.langchain import UnifiedReveniumCallbackHandler

# (a) Model attached once at startup; name the person inside the request handler.
def handle_request(user, question):
    @revenium_metadata(subscriber={"id": user.id, "email": user.email})
    def answer():
        return llm.invoke(question)
    return answer()

# (b) A model with no attach_to(); pass a handler with the call.
def handle_request(user, question):
    handler = UnifiedReveniumCallbackHandler(usage_metadata={"subscriber": {"id": user.id}})
    return bare_llm.invoke(question, config={"callbacks": [handler]})
```

With pattern (a), consume a stream inside the decorated function, or open the decorator around the iteration: a stream returned unconsumed (for example in a FastAPI `StreamingResponse`) is recorded without the decorator's fields.

Where the handler attached with `attach_to()` and `@revenium_metadata` both name a field (for example `subscriber`), the attached handler's value is used, on every model.

Combining a startup `attach_to()` with a per-call handler on the same model records each call once, with the attached handler's fields; the per-call handler's fields are used only when the attached handler has none.

**Embeddings models:** LangChain embeddings models run no callbacks, so `wrap()` and `attach_to()` return the model unchanged, attach nothing and log one warning per model class. Wrapping a chain or other runnable that has no callbacks field also attaches nothing and logs a warning; wrap the model inside it instead. `OpenAIEmbeddings` and `AzureOpenAIEmbeddings` calls are still recorded by the OpenAI middleware, without the `usage_metadata` given to `wrap()`; tag them with `@revenium_metadata` on the function that makes the call:

```python
from revenium_middleware import revenium_metadata
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

@revenium_metadata(organization_name="AcmeCorp", agent="support_agent", trace_id="session-123")
def embed(texts):
    return embeddings.embed_documents(texts)
```

Embeddings from vendors other than OpenAI, Azure OpenAI and Ollama through LangChain, such as `MistralAIEmbeddings`, are not metered (see [Calls the SDK does not meter](#calls-the-sdk-does-not-meter)).

**OpenAI Decisions, model router and auto mode:** a request made through langchain-openai's `OpenAIDecisions`, including the one `OpenAIModelRouterMiddleware` and `OpenAIAutoModeMiddleware` make on every agent run, is recorded with its input and output tokens when the handler reaches that run: pass it in the run's `config={"callbacks": [handler]}` (for example `agent.invoke(inputs, config={"callbacks": [handler]})`), or give the middleware a configured `OpenAIDecisions(...).with_config(callbacks=[handler])`. A handler attached to the chat model with `attach_to()` or `wrap()` does not see these requests. Decisions requests are not checked against spending caps, and a direct `client.decisions.create` call on the OpenAI client is not metered.

Credentials come from the standard environment variables (`REVENIUM_METERING_API_KEY`, `REVENIUM_METERING_BASE_URL`) or `revenium_middleware.initialize_metering(api_key=..., base_url=...)`. `revenium_middleware.configure()` sets the key for tool events only, so it does not meter LangChain calls.

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
from revenium_middleware import BudgetExceededError
from revenium_middleware.griptape import ReveniumDriver, enable_budget_errors

os.environ["REVENIUM_METERING_API_KEY"] = "your_revenium_key"
enable_budget_errors()  # once at startup, so a spending-cap refusal reaches your except block

agent = Agent(prompt_driver=ReveniumDriver(
    model="gpt-4o-mini",
    usage_metadata={"task_type": "demo"},
))
try:
    agent.run("Hello!")
except BudgetExceededError as e:
    print(f"Spending cap reached: {e.message}")
```

Force a provider with `force_provider="litellm"`, or wrap an existing driver with `ReveniumDriver(base_driver=...)`.

A wrapped OpenAI, Azure OpenAI, Claude or Ollama driver keeps every setting you gave it, including your own client, address, key, streaming and retry settings. A setting the Revenium driver cannot carry, such as an extra field on your own driver subclass, is named in a warning instead of being dropped silently.

To route Griptape through a LiteLLM gateway, pass `proxy_url=` (and `proxy_api_key=`) to `ReveniumDriver`, either as a base address or ending in `/chat/completions`. On that route Griptape tools are not sent and the gateway, not the SDK, records usage. The `LITELLM_PROXY_URL` environment variable alone does not change which provider a Griptape model uses, so `ReveniumDriver(model="gpt-4o-mini")` still calls OpenAI and records usage. Models that already route through LiteLLM (`force_provider="litellm"`, `gemini-*` and provider-qualified ids) still follow `LITELLM_PROXY_URL` through the LiteLLM driver's own setting. The proxy arguments are ignored, with a warning, when you wrap an existing driver.

**Spending caps:** Griptape turns any error inside a task into an error result, so without `enable_budget_errors()` an agent stopped by a cap returns the refusal text as its answer. With it, `agent.run(...)`, `Pipeline.run()` and `Workflow.run()` raise `BudgetExceededError`, in any thread. `run_stream` never raises: call `raise_if_budget_exceeded(agent)` after the stream ends, or after any run when you do not use the setup line. A refused call is never sent to the provider and is not retried. Nothing is installed on import: `enable_budget_errors()` sets Griptape's global observability driver, wrapping one you already set with `Observability.set_global_driver(...)` so it still sees every call, and `disable_budget_errors()` turns it off. If you use a `with Observability(...)` or `with GriptapeCloudStructure(observe=True)` block, call `enable_budget_errors()` inside each block, after it opens: leaving the block turns it off, and calling it before the block makes Griptape refuse the block. Pass your driver as `delegate=` only if you have already entered it or it has no enter or exit work.

Not covered: on the LiteLLM gateway route (`proxy_url=`) the gateway answers a cap with HTTP 429, which reaches you as `requests.HTTPError`, not `BudgetExceededError`. A refusal inside a tool (for example a vector-store embedding) reaches the model as the tool's output, not your `except`. The refusal is saved as the run's output in conversation memory, and Griptape still logs it at ERROR.

**Embeddings:**

```python
from revenium_middleware.griptape import ReveniumEmbeddingDriver

driver = ReveniumEmbeddingDriver(model="text-embedding-3-large")
```

**Provider-specific drivers:** for direct control, use `ReveniumOpenAiDriver`, `ReveniumAnthropicDriver`, `ReveniumOllamaDriver`, `ReveniumLiteLLMDriver` or `ReveniumOpenAiEmbeddingDriver` — each subclasses the corresponding Griptape driver and accepts a `usage_metadata` dict (see [Metadata Fields](#metadata-fields)).

**Migrating from `revenium-griptape`:** the standalone package is deprecated — install the `griptape` extra and change `from revenium_griptape import ReveniumDriver` to `from revenium_middleware.griptape import ReveniumDriver`. All driver class names are unchanged. One behaviour difference: the old package called `load_dotenv()` automatically at import time; the SDK never mutates your environment on import, so if you keep credentials in a `.env` file, call `load_dotenv()` yourself before creating a driver.

---

## Metadata Fields

Add business context to a call the SDK records by passing a `usage_metadata` dictionary. All fields are optional. The calls listed in [Calls the SDK does not meter](#calls-the-sdk-does-not-meter) record nothing from it; on LiteLLM's image generation, image edit, transcription, video generation and search calls, after `revenium_middleware.litellm` is imported, the SDK removes it before the request is sent, so it never reaches the provider.
These fields are shortened with a WARNING when longer than Revenium stores, so the call is still
recorded: `trace_id`, `task_type`, `agent`, `subscription_id`, `parent_transaction_id`, `environment`,
`region`, `transaction_name`, `operation_subtype`, `subscriber.id` and `subscriber.email` at 255
characters, and `product_name` at 224. Other text fields are sent as given.

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
Revenium backend rule is at most 16 characters, matching `^[A-Za-z0-9_-]+$`; a
value outside it is left off the record with a WARNING, and the call is still
recorded with its tokens and cost. It is distinct from the reported reasoning token count,
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
| `credential_alias` | `REVENIUM_CREDENTIAL_ALIAS` | Name of the provider API key used for the call, recorded as the subscriber's credential name | See usage and spend per provider key on calls with no subscriber id or email; a credential name or value you set on the subscriber wins |
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

**Note:** `operation_type` (e.g., `CHAT`, `EMBED`, `TOOL_CALL`) and `operation_subtype` (e.g., `function_call`, `streaming`) are automatically detected by the middleware and cannot be overridden. To label a function's calls, use `metadata={"task_type": "..."}` on `@revenium_meter` or a `transaction_name`.

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

The credential alias is stored as the subscriber's credential name (`subscriber.credential.name`). A call with no subscriber id or email is recorded under a credential named after the alias. A call that names a subscriber, by id or email, keeps that subscriber's own credential, and the SDK does not send the alias on it, so different subscribers are never linked to one shared credential. A credential name or value you set on the subscriber yourself (`subscriber.credential`, or `subscriber_credential_name` / `subscriber_credential`) wins over the alias in what the SDK sends. If your team uses strict ingestion, register the alias as a credential first, or calls that carry it can be refused.

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
- **API-level override**: `usage_metadata` passed directly to API calls always takes precedence over decorator metadata; a `None` or empty per-call value leaves the decorator's value in place
- **Async and streaming support**: Works with sync and async functions, and with generator and async generator functions such as streaming endpoints. In a generator, the metadata applies to the calls the generator body makes, not to code that handles each yielded item. The same holds for a streaming function behind a retry, tracing or caching decorator, and for a plain or async function that returns a generator
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

Calls outside a `@revenium_meter` function are not metered, but they still accept `usage_metadata`: the middleware removes it before the request reaches the AI provider. This covers LangChain models carrying the Revenium callback and raw boto3 Bedrock calls with `REVENIUM_BEDROCK_TRANSPORT=1`. A stream, through LangChain or raw boto3, is judged when it starts, so a stream started inside a metered function is recorded even if it is read to the end outside it. Spending caps (`REVENIUM_CIRCUIT_BREAKER_ENABLED=true`) still apply to calls outside a metered function, through every provider middleware. Nested `@revenium_meter` functions keep the outer function metered after the inner one returns. A decorated generator or async generator function, such as a streaming endpoint, has the calls its body makes metered and tagged. So does one that also carries another decorator listed below the Revenium ones, such as a retry, tracing or caching decorator, and a plain or async function that returns a generator. A generator handed back inside another object, such as `StreamingResponse(gen())`, is not seen: decorate the generator function itself.

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

**Worker threads:** The metering mark of `@revenium_meter` and the tags of
`@revenium_metadata` follow `await`, asyncio tasks and `asyncio.to_thread`, but
not `ThreadPoolExecutor`, `threading.Thread` or `loop.run_in_executor`: those
start the thread in an empty context. AI calls made on such a thread are not metered when selective metering
is on, and are recorded without the decorator's tags when it is off.

To carry them, wrap the thread's target with `carry_context` inside the decorated
function, each time you hand work to a thread:

```python
from concurrent.futures import ThreadPoolExecutor
from revenium_middleware import carry_context, revenium_meter, revenium_metadata

@revenium_meter()
@revenium_metadata(task_type="premium-feature", organizationName="PremiumTier")
def summarize_all(docs: list[str]) -> list[str]:
    with ThreadPoolExecutor() as pool:
        return list(pool.map(carry_context(summarize), docs))
```

`carry_context` is not a decorator: a wrap made outside the decorated function
carries nothing. With selective metering on, a thread started outside any
decorated function is never metered through it; with it off, such calls are
still metered, without the decorator's tags. Wrapping an `async def` carries
nothing either, because its coroutine runs later on the event loop, outside the
copied context: wrap the sync function the thread runs instead, or copy the
context around the awaited call. It also carries a `metering_key()` or `idempotency_key()`
block in force where it is called. The first time selective metering skips a
call, the SDK logs one INFO line saying so.

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

When a decorated tool raises, the event is sent with `success: false` and the exception's class (for example `HTTPStatusError`) as `errorMessage`, and the exception is re-raised. The message itself often holds request URLs, keys or personal data, so it is sent only when `REVENIUM_CAPTURE_ERROR_MESSAGES=true`, as `<Class>: <message>` capped at 2048 characters, with `[REDACTED]` in place of card numbers, every key shape prompt capture masks (plus `pk-` and `Token` values), JSON Web Tokens, query-string values, `Bearer` and `Basic` tokens, URL user names and passwords, values under labels ending in a sensitive name (`password`, `secret`, `token`, `api_key`, `authorization` and the like) and emails. An exception raised from a provider's HTTP error (an OpenAI, Anthropic, LiteLLM, google-genai, Perplexity or Ollama error, or an exception raised while handling one) can quote the request, prompt included, so its message is sent only when `REVENIUM_CAPTURE_PROMPTS=true` as well; otherwise the class alone is sent.

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

**Transaction ids:** each tool event gets its own transaction id, so every tool call is recorded. A `transaction_id` you set through `context()`, `set_context()` or `report_tool_call()` groups tool calls: it is sent as the event's `traceId`, or as `usageMetadata.callerTransactionId` when you also set a `trace_id`. A retry the SDK makes itself (an HTTP retry, or a replay from the delivery buffer) reuses the event's id and is stored once. Calling `report_tool_call` twice records two events, even with identical arguments.

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
| `system_prompt` | System prompt content | From `system` parameter / system message; for the OpenAI Responses API, `instructions` (else the first `system` or `developer` input item) |
| `input_messages` | User/assistant messages as JSON | From `messages` parameter, or the Responses API `input` |
| `output_response` | Assistant's response content | From response content blocks, or the text of a Responses API reply's messages |
| `prompts_truncated` | Truncation flag | Set to `true` if any field exceeded its maximum length |

Each field has a maximum length of **50,000 characters**. If exceeded, it's truncated with a `...[TRUNCATED]` marker. `input_messages` is always stored as valid JSON: when the messages do not fit, the longest texts are shortened first, each ending in the marker, and if that is not enough the oldest messages are left out.

Credentials in these three fields are masked before they leave your process, both in the records the provider integrations send with prompt capture on and in the text values of `systemPrompt`, `inputMessages` and `outputResponse` sent with `AgenticOutcomeClient.emit_completion`: keys starting `sk-`, `sk-proj-`, `sk-ant-`, `pplx-`, `xai-`, `gsk_`, `AIza`, `AKIA`, `ghp_`, `ghs_`, `gho_`, `ghu_`, `ghr_`, `github_pat_`, `hf_`, `rev_sk_`, `rev_mk_` and `hak_`, Slack tokens (`xoxb-`, `xoxp-` and the other `xox?-` forms), the secret half of a fal key (`<key id>:<secret>`), PEM private key blocks (`-----BEGIN PRIVATE KEY-----` and the RSA, EC and OpenSSH forms), card numbers, JSON Web Tokens, `Bearer` tokens, `Basic` credentials, the password in an address such as `postgres://user:password@host`, the signature of a presigned S3, Google Cloud Storage or Azure link, and the value after any label ending in `password`, `passwd`, `pwd`, `secret`, `secrets`, `token`, `api_key`, `credential` or `credentials`, or ending in `key` and naming a secret, private, access, API, auth, signing, client, session, master, encryption, account or storage key (so `AWS_SECRET_ACCESS_KEY=` and `private_key:` are masked, `password_hint:` and `max_tokens:` are not). The key's prefix, a fal key's id, a link's path and other parameters, and an address's user and host are kept; the secret is replaced with `***REDACTED***` (`***REDACTED_JWT***` for a JSON Web Token). A key glued to the letters or digits before it is not recognized.

Only signed JSON Web Tokens are masked to `***REDACTED_JWT***`, wherever the SDK masks them: three parts joined by dots, the first containing `eyJ` and the second starting with it. An encrypted five-part token and an unsigned token whose last part is empty are sent as written.

For LiteLLM, capture covers client mode chat calls (`completion`, `acompletion` and their streams), with the same fields and truncation as the OpenAI integration. Embeddings are not captured, and the LiteLLM gateway plugin does not capture the prompts or replies of the calls it proxies.

### Maximum Length

```bash
export REVENIUM_MAX_PROMPT_SIZE=20000
```

`REVENIUM_MAX_PROMPT_SIZE` sets the maximum length for every integration, as in the Go and Node SDKs. It must be a positive whole number; any other value is ignored with a warning. A value below 200 is raised to 200, the smallest length that still holds the `...[TRUNCATED]` marker, with a warning. Gemini and Vertex AI also read `REVENIUM_MAX_PROMPT_LENGTH`, and `REVENIUM_MAX_PROMPT_SIZE` wins when both are set. For Gemini and Vertex AI the length is capped at 1,000,000 characters; this cap is Python only.

The SDK reads these settings once, when it is imported, so call `load_dotenv()` before importing it.

The Go and Node SDKs also read `REVENIUM_ORGANIZATION_ID`, `REVENIUM_ORGANIZATION_NAME` and `REVENIUM_PRINT_SUMMARY`; the Python SDK does not.

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

On the OpenAI Responses API, capture covers `responses.create` and `responses.stream`, sync or async, streamed or not, through `with_raw_response`, and background calls. A Responses call captures only its own `instructions` and `input`: earlier turns OpenAI holds for you through `previous_response_id` or `conversation`, and a stored prompt template passed as `prompt=`, are not captured. `output_response` holds the text of the assistant's messages only; tool calls, reasoning and refusals are not captured, as for chat calls. Async `with_streaming_response` calls that are not streamed (`AsyncOpenAI` `chat.completions.with_streaming_response.create(...)` and `responses.with_streaming_response.create(...)`, read with `await response.parse()`) are not recorded today, so nothing is captured for them. Calls the SDK does not meter at all are not captured either; see [Calls the SDK does not meter](#calls-the-sdk-does-not-meter).

LangChain calls the SDK records through its callback (chat models no Revenium client integration meters, for example ChatMistralAI, ChatCohere, Fireworks, Hugging Face and langchain-google-vertexai models, and text-completion models) capture the system prompt, input messages and reply too, sync or async, streamed or not. A text-completion prompt is stored as one `user` message. Tool calls inside the history are not captured, though the OpenAI chat integration keeps them; a tool message keeps its `tool_call_id`.

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

Every provider middleware checks the cap: OpenAI, Azure OpenAI, Anthropic (including Claude on Bedrock and Foundry), Google Gemini and Vertex AI, Ollama, Perplexity, fal, the LiteLLM client and proxy, and the LangChain callback.

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
| `REVENIUM_CACHE_DIR` | unset | When set, the rule cache is mirrored to `<dir>/revenium_enforcement_rules.<scope>.json` so a restarted process doesn't fail-closed on the very first call. `<scope>` is a short hash of the enforcement base URL and `REVENIUM_TEAM_ID`, so services for different teams can share one folder and each loads only its own rules. Nothing is saved without `REVENIUM_TEAM_ID`. |

On/off and mode settings ignore case and surrounding spaces or line endings; a value the SDK does not recognise keeps the default and logs one warning naming the setting.

### Public API

Enforcement starts with the first call any provider middleware checks, for example:

```python
import revenium_middleware.openai  # auto-instruments openai
import openai

client = openai.OpenAI()
```

The pre-call check fires before every call these middlewares meter, including streamed, raw-response, beta and async variants and calls selective metering leaves unmetered:

- OpenAI and Azure OpenAI: chat completions, responses, embeddings and audio.
- Anthropic, including Claude on Bedrock (`AnthropicBedrock`, and raw boto3 calls with `REVENIUM_BEDROCK_TRANSPORT=1`), Claude on Vertex AI (`AnthropicVertex`) and Foundry: `messages.create`, `stream` and `parse`, and their beta twins.
- Google: google-genai content, embeddings, Imagen and Veo, and Vertex AI generation, chat, embeddings and Imagen.
- Ollama: chat, generate, embed and embeddings.
- Perplexity: the native client and the OpenAI-compatible client.
- fal: run, stream, submit and subscribe.
- LiteLLM: client completions and embeddings, and the proxy guardrail.
- LangChain: the callback checks every model unless its calls go through a Revenium middleware that is imported and checks them itself (OpenAI and Azure OpenAI, including DeepSeek, xAI, Together, Cerebras, Upstage and `ChatOpenAIMantle`; Anthropic, including `ChatAnthropicBedrock`, `ChatAnthropicMantle` and `ChatAnthropicVertex`; Gemini through `langchain-google-genai`; Ollama; LiteLLM; Perplexity; Claude on Bedrock with the Bedrock transport). So Groq, Fireworks, Mistral, Cohere, Hugging Face, `langchain-google-vertexai` Gemini and any vendor not listed here are checked by the callback. So are LangChain's text-completion models (`OpenAI`, `AzureOpenAI` and other `BaseOpenAI` subclasses): they call the legacy completions endpoint, which no middleware checks. `AnthropicLLM` and `OllamaLLM` are checked by their middleware. A refusal there is logged by LangChain at WARNING before it reaches your code.

LangChain embeddings models run no callbacks: `OpenAIEmbeddings` and `AzureOpenAIEmbeddings` are checked by the OpenAI middleware, Ollama embeddings are checked by the Ollama middleware, embeddings from other vendors are not checked.

When the circuit breaker is disabled, it is a no-op. When enabled:

1. A daemon thread (`revenium-enforcement-poll`) starts on first use.
2. It polls `GET {REVENIUM_ENFORCEMENT_BASE_URL}/v2/api/ai/enforcement-rules/{REVENIUM_TEAM_ID}` every `REVENIUM_CB_POLL_INTERVAL_SECONDS` with the `x-api-key` header.
3. Rules are cached in-process. The pre-call check only reads that cache: it never makes a network request and never sleeps, so it is safe on an event loop (LiteLLM proxy, `AsyncOpenAI`). When the cache is more than 120 s old the check wakes the poller to refresh it early and carries on with the rules it has. Only one refresh runs at a time per process.
4. A refresh that fails (timeout, refused connection, 5xx, or any 4xx such as a 403 for a wrong key or team id) keeps the cached rules, logs a warning, and is not retried for one `REVENIUM_CB_POLL_INTERVAL_SECONDS`, however much traffic arrives meanwhile.
5. `204 No Content` is treated as "no rules configured" — the cache is cleared.

A rule with filters (a cap scoped to an agent, model, provider, subscriber, credential, task type, product or organization) blocks or warns only the calls that match all of them; a filter the SDK cannot evaluate, such as the model of a LiteLLM proxy request, keeps the rule applying.

A model filter also holds a vendor's own aliases for the model it names, because Revenium records those calls under the model the vendor serves; an alias is honoured only on a call that can reach that vendor (its own client, a LiteLLM route to it, or a bare LiteLLM model name), so an Ollama model that shares the name is not held. A Gemini `-latest` alias (`gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-pro-latest`) is held by a `MODEL IS` or `MODEL IN` cap on any version of the same tier, such as `gemini-3.8-flash`, since Google moves the alias to each new model of that tier; a cap on another tier, or on a TTS, audio or image variant, does not hold it. A substring model filter (`CONTAINS`, `STARTS_WITH`, `ENDS_WITH`) whose value is a full model name of the same tier, such as `CONTAINS gemini-3.8-flash`, holds the alias the same way. A substring filter whose value is only part of a name, such as `STARTS_WITH gemini-3.8`, holds it only when the alias text itself matches, because which model Google will serve is not known before the call. Google's other `-latest` names that do not follow `gemini-<tier>-latest` (`antigravity-preview-latest`, `gemini-2.5-flash-native-audio-latest`) are held only by a cap on that name. On Vertex AI, which records the alias itself, a cap on the alias name holds it as well. An xAI short name such as `grok-4.20` or `grok-code-fast-1` is held by a cap on the full model xAI answers with (`grok-4.20-0309-reasoning`, `grok-build-0.1`), for every name xAI listed when this SDK version shipped; a name xAI adds or repoints later is held only after an SDK update.

### Exception Contract

```python
from revenium_middleware import BudgetExceededError
```

When a tripped rule matches the current request, the middleware raises before the provider call is made. `revenium_middleware.openai` still exports the same class. All structured fields are populated when the server provides them:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable reason, e.g. `"Request blocked by Revenium enforcement rule: monthly-gpt4-cap"` |
| `rule_name` | `str \| None` | Server-side rule name |
| `current_value` | `float \| None` | Current metric value at the time of the block |
| `threshold` | `float \| None` | Configured limit |
| `resets_at` | `str \| None` | ISO-8601 timestamp the rule next resets |
| `rule_id` | `str \| int \| None` | Server-side rule identifier |
| `deny_message` | `str \| None` | The message written for the rule, when it has one; also appended to `message` |

When the breached rule carries a deny message, `message` is the usual text followed by that message, for example `"Request blocked by Revenium enforcement rule: monthly-gpt4-cap. Ask the user before retrying."`, and `deny_message` holds the message alone, so your code can show it as written. Placeholders in it, such as `{spent}` or `{model}`, are shown exactly as written and not filled in by this SDK; read `current_value`, `threshold`, `rule_name` and `resets_at` from the exception for those values. A rule without a deny message keeps the usual text and leaves `deny_message` as `None`.

`BudgetExceededError` does **not** inherit from `ReveniumMiddlewareError`, and no provider middleware swallows it, so it reaches your `except` block, with two exceptions.

Griptape turns errors inside a task into an error result, so call `enable_budget_errors()` once at setup, or `raise_if_budget_exceeded(agent)` after a run; see [Griptape](#griptape).

`litellm.batch_completion` runs each item on its own thread and returns a refused item's `BudgetExceededError` as that item's entry in the result list instead of raising it, so check each entry:

```python
import litellm
from revenium_middleware import BudgetExceededError

batch = [[{"role": "user", "content": "Summarize A"}], [{"role": "user", "content": "Summarize B"}]]
results = litellm.batch_completion(model="gpt-4o-mini", messages=batch)
refused = [r for r in results if isinstance(r, BudgetExceededError)]
```

`litellm.batch_completion_models_all_responses` leaves a refused model out of its result list without raising.

```python
from revenium_middleware import BudgetExceededError
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

By default (`REVENIUM_CB_FAIL_MODE=open`) enforcement failures never propagate to user code and never delay it. If the rule fetch errors (network, 5xx, auth), the previous in-memory cache is preserved, a warning is logged, and the next attempt waits one poll interval. If there is no cache yet — the first moments after a process starts, or the whole of an outage that began before the first successful fetch — enforcement behaves as if no rules are configured and the request continues immediately; it does not wait for the fetch.

Set `REVENIUM_CB_FAIL_MODE=closed` to refuse calls until at least one rule fetch (or `REVENIUM_CACHE_DIR` snapshot) succeeds. The refusal is immediate too: a call made before the first fetch completes raises `BudgetExceededError` rather than waiting for it. Pair with `REVENIUM_CACHE_DIR` so a process restart loads the last-known rules rather than refusing every call until the first poll completes.

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

#### Metering Several Customers in One Process

`initialize_metering()` sets one key for the whole process. The record of a non-streamed
OpenAI (chat, Responses or embeddings), Anthropic `messages.create` (including
Claude on Bedrock through the Anthropic client), LiteLLM, Ollama or Perplexity
call is filed under the key that is current when the call returns. The
exceptions are an Anthropic `with_streaming_response` call and a Perplexity
response read raw or lazily, which keep the key from when they started. Every
other record, including streamed calls, `@meter_tool` tool events, and Google
Gemini, Vertex AI, OpenAI audio, fal.ai and raw boto3 Bedrock calls, keeps the
key that was current when it started; one started while metering was off is not
recorded. A call that fails with a provider error is also filed under the key
from when it started. A call made through LangChain, CrewAI or Griptape follows
the rule of the provider client it uses, except that a LangChain call recorded
by the Revenium callback rather than by a provider client, such as
`wrap(ChatOllama(...))` without the Ollama middleware, keeps the key from when the
run started. So switching keys with
`initialize_metering()` while another thread or asyncio task still has a call
or a metered tool running can send that record under the new key.

To meter several of your own customers at once, give each thread or asyncio
task its own key with `metering_key()`:

```python
import revenium_middleware

with revenium_middleware.metering_key("hak_customer_key"):
    client.chat.completions.create(...)
```

The key applies to the usage records and tool events of calls made inside the
block, on that thread or asyncio task only: a thread started inside the block
does not inherit it, while an asyncio task created inside it does. Wrap the
thread's target with `carry_context` to carry the key.
`base_url` is optional and defaults to the address set with
`initialize_metering()` or `REVENIUM_METERING_BASE_URL`. An empty or malformed
key raises `ValueError`.

- A key set with `configure(api_key=...)` still wins for tool events, and those events go to the address set with `configure(metering_url=...)`, else the process-wide address, never to the block's `base_url`. Without a `configure()` key, tool events in the block go out with the block's key and the block's address together, even when the block sets no `base_url` (the block's key then goes to the process-wide address); `configure(metering_url=...)` alone does not redirect them.
- Spending caps follow the process-wide key (the environment or
  `initialize_metering()`) and `REVENIUM_TEAM_ID`, never the key of a
  `metering_key()` block. A service that sets keys only with `metering_key()`
  has no process-wide key, so it gets no spending caps, and with
  `REVENIUM_CB_FAIL_MODE=closed` every call is refused.
- Outcomes and jobs are not covered: they need a write key (`rev_sk_`).
- Delivery pauses for the whole process while Revenium is not answering at
  any one address (see [Metering Delivery and Timeouts](#metering-delivery-and-timeouts));
  the records held back meanwhile are replayed later under their own key.

### Metering Delivery and Timeouts

Usage records are never sent on your request path. A metered call puts its
record on a bounded in-process queue and returns at once; a fixed pool of
background threads sends queued records to Revenium. The number of threads
does not grow with traffic, so a slow or unreachable metering endpoint costs
a few idle threads and the queue's memory, never a thread per call.

| Variable | Default | What it sets |
|----------|---------|--------------|
| `REVENIUM_METERING_WORKERS` | `32` | Background threads that send records |
| `REVENIUM_METERING_QUEUE_SIZE` | `1000` | Records that can wait for a thread |
| `REVENIUM_METERING_TIMEOUT_SECONDS` | `10` | Seconds one attempt waits for a response |
| `REVENIUM_METERING_CONNECT_TIMEOUT_SECONDS` | `5` | Seconds one attempt waits for a connection |
| `REVENIUM_METERING_MAX_RETRIES` | `2` | Retries after the first attempt |

The two timeouts also apply to tool events, including those sent with
`emit_tool_event`, which are sent once without retries, and a replay from the store-and-forward buffer is a single attempt
that waits at most `REVENIUM_METERING_TIMEOUT_SECONDS`.

Each thread sends one record at a time, so the threads keep up with about one
record per thread per round trip to Revenium. The 32 default threads send
about 100 records a second at a 300 ms round trip and keep up with 30 a
second until a round trip takes about a second. Measured against a local
endpoint answering in 250 ms and 300 ms, 30 calls a second for five minutes
delivered every record within 0.35 seconds of the call, and so did 60
records a second at 250 ms (a service that sends two records per call).
From one client in Brazil to a US East Revenium API host, a request on an
open connection took 113 ms (p50) and 123 ms (p99), and one that also opened
the connection 333 ms and 462 ms;
measure from your own region before sizing for a slower path. Idle, the 32
threads cost about 1.4 MB of memory. Raise `REVENIUM_METERING_WORKERS` for a
service that meters more than 30 records a second against a slow round trip;
the queue absorbs short bursts. When the queue is full, new records go
straight to the store-and-forward buffer below instead of waiting, and
`get_buffer_stats()["total_overflowed"]` counts them; one warning marks the
start of each such episode and one, with a count, its end.

When Revenium stops answering, the threads stop waiting on it. After three
records in a row fail with a retryable error, new records go straight to the
store-and-forward buffer without being sent, and one record every 5 seconds
is still sent to find out whether Revenium is back. The first one that
succeeds resumes sending and starts replaying the buffer at once. A warning
marks each change. Tool events from `@meter_tool` and `report_tool_call` are
not held back: `configure()` can send them to another endpoint, so they keep
being sent, and the buffer replays them even while the AI records ahead of
them are still failing.

At exit, queued records get the first share of the shutdown budget (see
[Shutdown](#shutdown)).

### Delivery Resilience (Store-and-Forward)

Metering events that still fail after the client's own retries (network
outages, 5xx, rate limiting), and the records held back while Revenium is not
answering, are not lost: they are held in a bounded in-memory buffer and
replayed automatically in the background every 30 seconds, or as soon as a
delivery succeeds again, reusing each event's original `Idempotency-Key` so
replays can never double-bill. A backlog replays 16 records at a time, about
50 a second at a 300 ms round trip next to your live traffic. Permanent
failures (401/403/404/422) are never buffered, and buffered events expire
after 24 hours. The buffer is drained on shutdown (see below).

The buffer holds up to 20,000 events, eleven minutes of an outage at 30
records a second. A typical LiteLLM record takes about 2.3 KB in memory, so a
full buffer holds about 45 MB; with `REVENIUM_CAPTURE_PROMPTS` each record
can also carry up to 50,000 characters per captured prompt field, so lower
`REVENIUM_BUFFER_MAX_SIZE` if you capture prompts. In a five-minute outage at
30 calls a second against the defaults, all 9,000 records were held and then
delivered within two and a half minutes of Revenium answering again, while
new traffic kept flowing. When the buffer is full the oldest
events are evicted and their usage is lost: one warning marks the first
eviction, the flush thread logs how many were evicted at most once per flush
interval, and `get_metering_status().evicted_count` and
`get_buffer_stats()["total_evicted"]` count them. Inspect it programmatically:

```python
from revenium_middleware import get_buffer_stats

print(get_buffer_stats())
# {'size': 0, 'max_size': 20000, 'total_buffered': 3, 'total_replayed': 3, 'total_evicted': 0, ...}
```

### Shutdown

The SDK does not install signal handlers: SIGTERM and SIGINT reach your
application or server exactly as they would without it. When the interpreter
exits normally, including after Ctrl+C, an `atexit` hook works through three
steps against one budget of `REVENIUM_SHUTDOWN_TIMEOUT_SECONDS` (5 seconds by
default), each using whatever time the previous ones left: it waits for the
background threads to send the records still queued, builds the records that
overflowed to the store-and-forward buffer, then flushes the buffer, oldest
first, up to 32 records at a time, one attempt per record. Once a usage record or a tool event fails
in a way worth retrying (unreachable, timed out, 429 or 5xx), the flush sends
no more records of that kind, leaving them unsent, and keeps sending the other
kind, since tool events can go to a different endpoint. Records still queued, being sent, unbuilt
or unsent at the end are dropped and logged once, as a warning with a count.

A process ended by SIGTERM's default action skips `atexit`, and with it that
final flush. uvicorn, and so the LiteLLM proxy, ends that way once its own
graceful shutdown has finished. For a script or worker with no SIGTERM handling
of its own, set `REVENIUM_INSTALL_SIGNAL_HANDLERS=1`: SIGTERM then flushes
first and ends the process the same way. A SIGTERM handler your code installed
before importing the SDK is called unchanged, and one installed afterwards
replaces the SDK's. The flush then happens only if that handler exits
normally: a handler that restores SIGTERM's default action and re-raises it,
as uvicorn does after its own shutdown, ends the process before `atexit` runs.

#### Worker processes

A `multiprocessing` child started with the fork or forkserver start method
(the Linux default) ends through `os._exit`, which skips `atexit`. The SDK
therefore also flushes a `multiprocessing.Process` child when its target
returns, with the same budget and the same warning. Both start methods fork
the child from a process that has imported the SDK, and a forked child starts
with its own metering queue, buffer and connections: it never sends records
its parent queued, and its warning counts only its own records.

Some worker shapes still need a call from you, or are not covered:

- `multiprocessing.Pool` workers flush when the worker process exits (after
  `pool.close(); pool.join()`, or at `maxtasksperchild`), not after each task.
  Leaving a `with Pool() as pool:` block calls `terminate()`, which kills the
  workers and can still lose their last records.
- A raw `os.fork()` worker, or an RQ work horse, that ends with `os._exit`
  must call `flush_metering()` at the end of each job.
- Celery's prefork pool (billiard) is not covered; call `flush_metering()` at
  the end of a task if its worker processes end that way.
- Under fork and forkserver, records made by threads still running after the
  target returns are not delivered.

```python
import revenium_middleware

def job():
    ...  # model calls
    left = revenium_middleware.flush_metering()  # waits up to REVENIUM_SHUTDOWN_TIMEOUT_SECONDS
    # left == 0: every record was sent; otherwise that many are still unsent
```

`flush_metering(timeout_seconds=None)` sends the records queued or held so
far and returns how many are still unsent. Metering carries on afterwards, so
a long-lived worker can call it after every job. A negative or non-finite
timeout raises `ValueError`.

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REVENIUM_METERING_BASE_URL` | `https://api.revenium.ai` | Revenium API endpoint |
| `REVENIUM_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `REVENIUM_CAPTURE_PROMPTS` | `false` | Enable prompt capture |
| `REVENIUM_CAPTURE_ERROR_MESSAGES` | `false` | Send the exception message, redacted and capped at 2048 characters, with failed `@meter_tool` events and auto-failed `JobContext` outcomes, and cut to 255 characters (keeping its "(HTTP <status>)" ending) on the record of a failed OpenAI, Anthropic, LiteLLM, Google, Perplexity or Ollama call; off sends the exception class only (with the HTTP status for a failed call). A provider HTTP error's message, which can quote the request, is sent only when `REVENIUM_CAPTURE_PROMPTS` is also on |
| `REVENIUM_SELECTIVE_METERING` | `false` | Only meter `@revenium_meter` decorated functions |
| `REVENIUM_TEAM_ID` | - | Team ID for cost lookups and outcome reporting (JobContext team resolution) |
| `REVENIUM_ENVIRONMENT` | - | Deployment environment (auto-detects from `ENVIRONMENT`, `DEPLOYMENT_ENV`) |
| `REVENIUM_REGION` | - | Cloud region (auto-detects from `AWS_REGION`, `AZURE_REGION`, `GCP_REGION`) |
| `REVENIUM_CREDENTIAL_ALIAS` | - | Provider API key name, recorded as the subscriber's credential name on calls with no subscriber id or email |
| `REVENIUM_TRACE_TYPE` | - | Workflow category identifier |
| `REVENIUM_TRACE_NAME` | - | Human-readable trace label |
| `REVENIUM_PARENT_TRANSACTION_ID` | - | Parent transaction ID for distributed tracing |
| `REVENIUM_TRANSACTION_NAME` | - | Human-friendly operation name |
| `REVENIUM_RETRY_NUMBER` | - | Retry attempt number |
| `REVENIUM_AGENTIC_JOB_ID` | - | Agentic job instance ID attached to all completions in the process (triggers backend job auto-creation; same format rule as `JobContext`) |
| `REVENIUM_AGENTIC_JOB_NAME` | - | Human-readable agentic job name |
| `REVENIUM_AGENTIC_JOB_TYPE` | - | Agentic job type category |
| `REVENIUM_AGENTIC_JOB_VERSION` | - | Agentic job version |
| `REVENIUM_WRITE_API_KEY` | - | Primary write-scope key (`rev_sk_`) for the agentic outcomes API (report/amend/history); falls back to `REVENIUM_OUTCOME_API_KEY` (deprecated), then `REVENIUM_METERING_API_KEY` |
| `REVENIUM_OUTCOME_API_KEY` | - | Deprecated fallback name for the write-scope key; used only when `REVENIUM_WRITE_API_KEY` is unset |
| `REVENIUM_PROFITSTREAM_BASE_URL` | same address as `REVENIUM_METERING_BASE_URL` | Agentic outcomes API base URL. Only needed when outcomes live at a different address than usage. By default it follows the `REVENIUM_METERING_BASE_URL` environment variable: a trailing `/meter` is removed and a path prefix is kept. `AgenticOutcomeClient` also follows a usage address set in code with `initialize_metering(base_url=...)` or a `metering_key(..., base_url=...)` block, for its jobs and outcomes as for its usage records, unless this variable or `profitstream_base_url=` sets an outcome address, or `meter_base_url=` names an address other than the one in `REVENIUM_METERING_BASE_URL`. |
| `REVENIUM_BEDROCK_DISABLE` | - | Set to `1` to stop Anthropic clients being detected as Bedrock; Foundry detection is unaffected. Raw boto3 calls metered with `REVENIUM_BEDROCK_TRANSPORT`, and LangChain's `ChatBedrock` and `ChatBedrockConverse`, are still recorded as AWS |
| `REVENIUM_BEDROCK_TRANSPORT` | - | Set to `1` or `true` to meter Claude models called through a raw `boto3.client("bedrock-runtime")` (`invoke_model`, `invoke_model_with_response_stream`, `converse`, `converse_stream`); other Bedrock models are never metered this way |
| `REVENIUM_BUFFER_MAX_SIZE` | `20000` | Store-and-forward buffer capacity (oldest events evicted when full) |
| `REVENIUM_BUFFER_FLUSH_INTERVAL` | `30` | Seconds between automatic replay attempts for buffered events |
| `REVENIUM_METERING_WORKERS` | `32` | Background threads that send usage records (see [Metering Delivery and Timeouts](#metering-delivery-and-timeouts)) |
| `REVENIUM_METERING_QUEUE_SIZE` | `1000` | Usage records that can wait for a sending thread; overflow goes to the store-and-forward buffer |
| `REVENIUM_METERING_TIMEOUT_SECONDS` | `10` | Seconds one attempt to send a usage record waits for a response |
| `REVENIUM_METERING_CONNECT_TIMEOUT_SECONDS` | `5` | Seconds one attempt waits for a connection |
| `REVENIUM_METERING_MAX_RETRIES` | `2` | Retries after the first attempt before a record moves to the buffer |
| `REVENIUM_SHUTDOWN_TIMEOUT_SECONDS` | `5` | Total seconds the exit hook may spend sending queued usage records, building overflowed ones and flushing the buffer |
| `REVENIUM_INSTALL_SIGNAL_HANDLERS` | `false` | Set to `1` to flush metering before SIGTERM's default action ends the process (see [Shutdown](#shutdown)) |

On/off and mode settings ignore case and surrounding spaces or line endings; a value the SDK does not recognise keeps the default and logs one warning naming the setting. `REVENIUM_DEBUG` and `REVENIUM_BEDROCK_DISABLE` are the exceptions: they ignore surrounding spaces or line endings, but `REVENIUM_DEBUG` accepts only `true`, `1` or `yes`, `REVENIUM_BEDROCK_DISABLE` accepts only `1`, and neither logs a warning for any other value.

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
| **Bedrock: AccessDenied** | Ensure `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` permissions; add `bedrock:GetInferenceProfile` for application inference profile ARNs with `REVENIUM_BEDROCK_TRANSPORT=1`, or with `AnthropicBedrock` and cost controls on, and `bedrock:GetProvisionedModelThroughput` for Provisioned Throughput ARNs with `REVENIUM_BEDROCK_TRANSPORT=1` |
| **Bedrock: requests go to Anthropic** | Verify AWS credentials: `aws sts get-caller-identity` |
| **Google: embeddings show 0 tokens** | Expected on the Gemini API, which returns no embedding token count; use google-genai with `vertexai=True` (Vertex AI) for counted embeddings |
| **Google: "No module named 'vertexai'"** | Install correct extra: `pip install "revenium-python-sdk[google-vertex]"` |
| **Vertex AI: authentication errors** | Run `gcloud auth application-default login` |
| **Ollama: connection errors** | Ensure Ollama is running: `ollama serve` |
| **LangChain: provider shows "unknown"** | Ensure you're using a supported LangChain LLM class |
| **Streaming errors** | Check provider credentials; middleware auto-falls back gracefully |

**Debug mode:** Set `REVENIUM_LOG_LEVEL=DEBUG` (or `REVENIUM_DEBUG=true`, which is the same for every integration) to see detailed provider detection, routing decisions, and metering payloads, with credentials, emails and prompt or reply text redacted. Token counts, ids, model names and attribution names still print.

**Stop Bedrock detection on Anthropic clients:** Set `REVENIUM_BEDROCK_DISABLE=1` to disable Bedrock auto-detection on Anthropic clients. It does not send your calls to Anthropic directly: an `AnthropicBedrock` call still goes to AWS and is billed by AWS, but is recorded as provider `anthropic` at the Anthropic price. Foundry detection is unaffected - a Foundry client is still labelled `Foundry`. Raw boto3 calls metered with `REVENIUM_BEDROCK_TRANSPORT` and LangChain's `ChatBedrock` and `ChatBedrockConverse` are still recorded as AWS.

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

`REVENIUM_DEBUG=true` is the same as `REVENIUM_LOG_LEVEL=DEBUG` for every integration, and takes precedence over `REVENIUM_LOG_LEVEL`. Both are read when `revenium_middleware` is first imported; a level your application sets on the `revenium_middleware` logger afterwards takes precedence.

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
