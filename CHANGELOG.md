# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.8] - 2026-06-04

### Fixed
- Prompt-cache token counts (cache reads and cache writes) are now correctly extracted and reported across providers instead of being metered as 0:
  - OpenAI completions now report cached prompt tokens for both standard and streaming responses, with streaming cache token extraction unified across response shapes.
  - Anthropic on AWS Bedrock now parses cache token counts from invoke and streaming usage events and propagates them to metering.
  - The LangChain integration now recognizes Anthropic's native cache token fields (`cache_read_input_tokens` / `cache_creation_input_tokens`), so cache-aware metering works when using LangChain with Anthropic models.
- The LangChain integration now finds token usage in more response shapes: usage stored on generation messages (`LLMResult.generations[..].message`) is picked up, usage lookups no longer stop at the first missing location, and object-style (non-dict) usage payloads are handled.

## [0.1.7] - 2026-05-28

### Added
- HMAC webhook verification helper (`revenium_middleware.webhooks.verify_signature`) for verifying webhook payloads signed by Revenium. Supports secret rotation by accepting multiple secrets, multiple signatures per request (RFC 7230 multi-value headers), configurable timestamp tolerance, and case-insensitive hex comparison. README includes a FastAPI verification example.
- Automatic `Idempotency-Key` generation for AI metering submissions. Duplicate retries of the same metered event are now safely deduplicated by the Revenium backend without any caller-side work. Callers can still override the key explicitly via the new `idempotency_key` context manager when they need to control the value (for example, to tie metering to an upstream request id).

### Changed
- All provider integrations (OpenAI, Anthropic, Google, Ollama, LiteLLM, Perplexity, fal.ai) now submit metered AI events through a single shared `submit_ai_event` wrapper. This unifies idempotency-key handling across providers and makes future header-level changes a single-point edit.
- `Idempotency-Key` is also inlined on the per-instance metering client used by the agentic-outcomes API, so outcomes submissions get the same dedup semantics as completion metering.

### Security
- `verify_signature` now rejects empty-string secrets at the type guard. An empty secret would otherwise be used as a zero-byte HMAC key, which any caller could forge against trivially. Configuration mistakes that produce empty secrets now fail closed.

### Fixed
- `MeteringThread` now correctly propagates Python `contextvars` from the calling thread, so metadata set via `set_idempotency_key`, decorators, or context managers reaches the background metering worker reliably.
- `set_idempotency_key('')` and an empty `idempotency_key=` argument now raise `ValueError` instead of silently bypassing dedup. An `Idempotency-Key` passed through `extra_headers` is similarly rejected to avoid two competing sources of truth.
- The `Idempotency-Key` header guard in `submit_ai_event` is now case-insensitive, so callers passing alternate casings (e.g. `idempotency-key`) through `extra_headers` still hit the guard rather than silently shadowing the wrapper's value.
- Deprecated-field log warnings (`organizationId` / `productId` aliases) are now deduplicated per field pair so high-volume applications using legacy aliases no longer get a `logger.warning` flood on every call.

## [0.1.6] - 2026-05-22

### Removed
- Removed terminal usage-summary printer (opt-in `REVENIUM_PRINT_SUMMARY`) — undocumented dev feature; dashboard remains the source of truth for cost confirmation.

### Changed
- Refresh OpenAI model name in README and examples from `gpt-5.5` / `gpt-5.5-mini` to `gpt-4o-mini` so copy-paste examples resolve against current OpenAI catalogues.
- Refresh OpenAI getting-started example to use a realistic prompt + token count so the first metered call produces a visible non-trivial cost.
- Migration note: `organizationId` and `productId` field aliases remain accepted by the SDK as a backward-compat input convenience (with `DeprecationWarning`), but the backend has stopped silently mapping them and the canonical wire shape is `organizationName` and `productName`. New examples and documentation use the canonical names.

## [0.1.5] - 2026-05-21

### Added
- `AgenticOutcomeClient` and `AgenticOutcomeSettings` for emitting agentic-job outcomes (`CONVERTED`, `DEFLECTED`, `ESCALATED`) alongside completion and tool-event metering.
- `examples/agentic_outcomes/` example pack — runnable demo scripts (sales, coding, customer support) showing how to emit outcomes from real agent workflows.
- Automatic team-id discovery from the configured Revenium API key. Setting `REVENIUM_TEAM_ID` is no longer required when the SDK can resolve the team from the API key.

### Changed
- Outcome values are now reported as `float`, preserving fractional dollars (previously truncated by `int()`).
- Renamed `ReveniumCostLimitExceeded` to `BudgetExceededError` to align the exception name across Python, Node, and Go SDKs. The old name remains as a deprecated alias, so existing `except ReveniumCostLimitExceeded:` code keeps working.
- Refreshed example model references to match the current verified Revenium model catalog.

### Fixed
- Bedrock adapter now correctly maps `claude-opus-4-7`.
- Agentic-outcomes example scripts now reject `--count 0` instead of failing with a `ZeroDivisionError`.

### Documentation
- README now documents the agentic outcomes API.
- Cost-control terminology note ported from the Go SDK README for consistency across SDKs.

## [0.1.4] - 2026-05-08

### Added
- Server-side cost controls with circuit-breaker enforcement. The SDK now consults Revenium cost-control policies before each request and short-circuits calls that would exceed configured spend or usage limits.
- Documentation for configuring server-side cost controls in the README.

### Fixed
- Suppressed spurious `REVENIUM FAILURE` log entries when no API key is configured. Metering callers now safely no-op instead of dispatching against an uninitialized client.

## [0.1.3] - 2026-04-28

### Added
- API key prefix validation at SDK initialization: keys that are explicitly set but do not start with `hak_` or `rev_` now raise `ValueError` immediately instead of silently producing failing metering requests.
- camelCase aliases for metadata fields `trace_id`, `task_type`, `subscription_id`, `agent`, and `response_quality_score`. Previously these values were silently dropped when sent as `traceId`, `taskType`, and so on.
- Agentic job tracking via metadata fields `agentic_job_id`, `agentic_job_name`, `agentic_job_type`, and `agentic_job_version` across every provider integration.
- LiteLLM proxy support for agentic job fields through `x-revenium-agentic-job-id` and related HTTP headers.

### Fixed
- OpenAI summary printer now sends `Authorization: Bearer <key>`, bringing it in line with the Anthropic, Google, and LiteLLM summary printers.
- Bedrock adapter no longer sends deprecated metadata field names on the wire.
- `REVENIUM_SELECTIVE_METERING` is now honored by every provider wrapper. Previously some wrappers ignored the flag and metered all calls.
- Async client interception now covers `AsyncCompletions`, `AsyncMessages`, and `AsyncEmbeddings`, so async calls are metered consistently with sync calls.
- Perplexity calls routed through the OpenAI SDK are no longer double-metered. Perplexity URLs are skipped by the OpenAI wrapper, and a patch registry prevents both wrappers from attaching to the same client.
- Streamed completions are no longer double-metered through the async stream wrapper.
- Concurrent shutdowns no longer race on internal thread bookkeeping.
- A missing `response.usage` no longer raises; the middleware reports zero usage and continues.
- Anthropic: corrected `stop_reason` mapping, metadata sanitization, and initialization guard.
- OpenAI: trace fields are now propagated through Responses API streaming; user-provided `kwargs` are no longer mutated; Azure configuration race resolved.
- LiteLLM: streaming and proxy-async paths metered correctly; token double-counting corrected.
- Google: per-provider locking around shared state and improved streaming error handling.
- fal: `Config` inheritance corrected; response objects are no longer mutated.
- Ollama: response objects are no longer mutated by the middleware.
- Several robustness improvements across providers: narrower exception handlers, client cache eviction, kwargs sanitization, `response.id` fallback, dynamic `CAPTURE_PROMPTS` evaluation, and dependency version pinning.

### Changed
- Metadata extraction is now centralized with a standardized `snake_case > camelCase > deprecated_snake > deprecated_camel` precedence across all providers.
- Perplexity migrated from the deprecated `organization_id` / `product_id` fields to the centralized extraction pipeline.
- When no API key is configured, the metering client is `None` and a clear warning is logged at initialization.

## [0.1.2] - 2026-04-09

### Added
- fal.ai middleware support with endpoint routing, per-type field extraction, trace fields, media type detection, and model normalization
- Agentic job tracking fields (`job_id`, `job_name`, `squad_id`, `squad_name`) to all provider getting started examples

### Changed
- Comprehensive README rewrite with full provider guides, updated copyright year to 2026

## [0.1.0] - 2026-03-05

### Added
- Initial release of `revenium-python-sdk` — the unified Revenium Python SDK
- Unified middleware for all AI providers in a single package:
  - OpenAI (including Azure OpenAI)
  - Anthropic (including Bedrock)
  - Google (Gemini via Google AI and Vertex AI)
  - Ollama
  - LiteLLM (client and proxy)
  - Perplexity (OpenAI SDK and native SDK)
- LangChain integration support
- Core metering functionality with asynchronous processing
- Decorator support (`@revenium_meter`, `@revenium_metadata`)
- Context management utilities for thread-safe metadata tracking
- Tool metering via `meter_tool` decorator and `report_tool_call`
- Selective metering via `REVENIUM_SELECTIVE_METERING` environment variable
- Configurable logging with `REVENIUM_LOG_LEVEL`

