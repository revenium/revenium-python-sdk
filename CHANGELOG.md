# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
