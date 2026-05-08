# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Notes
- This package supersedes the individual provider packages (`revenium-middleware-openai`, `revenium-middleware-anthropic`, etc.)
- The import path remains `revenium_middleware` for backward compatibility
- Requires `revenium_metering>=6.8.2`
