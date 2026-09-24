# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-09-24

### Added
- AI completions can now carry prompt, speed-mode and subagent attribution. `create_completion` (sync and async) accepts `prompt_id`, `prompt_length`, `query_source`, `speed` and `subagent_type`, and every provider integration that meters completions reads the same five from `usage_metadata` under their snake_case or camelCase names (`promptId`, `promptLength`, `querySource`, `speed`, `subagentType`), so a coding-assistant call can be grouped by the prompt that caused it, split into fast-mode and normal-mode spend, and attributed to the subagent that made it. Values are sent as given; the API owns the accepted formats. Unset fields are left off the request, so existing integrations send exactly what they did before. They apply to completions only: passing them on an image, video or audio call changes nothing. The LiteLLM proxy, which takes attribution from `x-revenium-*` headers, does not read them yet.
- You can now ask the enforcement API about **one** cost-limit rule instead of reading the whole team's. `revenium_middleware._core.fetch_enforcement_rule(rule_id)` sends the server's new `ruleId` filter on `GET /v2/api/ai/enforcement-rules/{teamId}` and returns that rule as the server compiled it, or `None` when the team has no such compiled rule (a disabled rule is never compiled) or the read could not be completed. It is an inspection call: it neither reads nor writes the cache the pre-call check evaluates, and it is never on the path of a metered call. The background poller and the stale-cache refresh keep reading the whole team, unchanged and byte-for-byte — narrowing them would drop the department-budget maps the server computes team-wide and attaches to that read, and department budgets would silently stop blocking anyone. An empty `rule_id` raises `ValueError` rather than quietly reading the whole team.
- You can now see **who** a cost-limit rule actually covers. `revenium_middleware._core.fetch_enforcement_rule_roster(rule_id)` reads `GET /v2/api/ai/enforcement-rules/{teamId}/roster` and returns the people (or departments) the rule is measuring — each with their spend, their cap, their band and their last enforcement event — beside the whole roster's blocked, warned and under counts. That is the answer to "why was this caller blocked, and who else is this rule measuring?", which previously required reading the whole team's compiled payload and reassembling it by hand. `page`, `size`, `search` and `band` are passed to the server, which does the filtering, sorting, banding and paging; the band counts always describe the whole roster, so narrowing to one band cannot zero the other two. The roster is fetched on demand and never cached, so a stale roster cannot outlive the rule it describes, and nothing on the enforcement hot path reads it. It shares the rule fetch's retry, `Retry-After` budget, refresh cooldown and fail-open posture rather than issuing a bare request, so a throttled enforcement API sees one control-plane request, not two. Returns `None` when the rule has no compiled reading yet, the team has no such rule, or the read could not be completed.
- A Claude Code call that both Claude Code and your LiteLLM proxy report to Revenium can now be stored once instead of twice. `ReveniumGuardrail` mints one identifier per proxied Anthropic messages request, returns it to the client as the `request-id` and `x-revenium-transaction-id` response headers, and reports that same value as the call's transaction id. Claude Code copies `request-id` onto the usage record it reports itself, so the two records carry one identifier and Revenium's team-scoped duplicate check keeps one of them. On by default; set `REVENIUM_LITELLM_SHARED_CALL_ID=false` to opt out, and new calls go straight back to two records when you do; records already merged stay merged. Three preconditions: LiteLLM 1.93.0 or newer (the floor of the `litellm-proxy` extra, which is what installs the guardrail), a guardrail `mode` that includes `pre_call` (that is where the identifier is minted), and both sides reporting under keys of the same Revenium team. Opted out, every stored value is unchanged on every route. On a LiteLLM without the response-header hook, or under a `mode` that never runs the pre-call hook, the flag changes nothing and the guardrail logs one warning at startup naming the variable and the fix. Only the guardrail mints: a proxy still on the deprecated `MiddlewareHandler` callback alone keeps reporting two records, and a proxy running both without `default_on: true` sees its existing double count hidden rather than fixed, so delete the `litellm_settings.callbacks` entry. Other routes are untouched, so the provider's own `request-id` on the pass-through route is never overwritten, and a caller cannot decide their own transaction id by putting one in the request body. Point Claude Code at the proxy root (`ANTHROPIC_BASE_URL=<proxy>`) so it calls `<proxy>/v1/messages`: LiteLLM's Anthropic pass-through at `<proxy>/anthropic/v1/messages` is out of scope for the mint, and a proxy serving that route keeps counting the call twice with nothing in the log to say so.
- A LiteLLM proxy can now enforce a budget, not just report on one. `revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail` is a LiteLLM `CustomGuardrail` that blocks a call before it reaches the provider when a Revenium cost-limit rule is tripped, and meters it afterwards — successes, failures and streamed responses alike. Configure it under `litellm_settings.guardrails` with `mode: ["pre_call", "post_call"]` and `default_on: true`; see the README's LiteLLM "Proxy Mode". The pre-call decision is the SDK's own circuit breaker (`check_enforcement`, including the department/org-unit budget logic from 0.7/0.8), so a proxy enforces exactly what every other Revenium integration enforces, and a blocked caller receives HTTP 429 with a body naming the rule, its threshold, the caller's balance and when the window resets. Enforcement fails open — an unreachable or misbehaving enforcement path lets the call through — and metering is non-disruptive: the post-call hook runs in-band, so nothing it does can turn a successful LLM call into an error for the client. Metered rows carry the same `x-revenium-*` header attribution the callback records (with virtual-key metadata as the fallback), plus cache-token counts, `effort`, and `agenticJob*` tags, and are marked `middleware_source: "GUARDRAIL"`. Requires the `litellm-proxy` extra and Python 3.10+.
- Streamed proxied calls are metered with their assembled totals. LiteLLM hands the guardrail the complete response after a stream finishes, so a streamed call records real token counts and `is_streamed: true` rather than nothing. The vendor contract this depends on — that a guardrail defining no `async_post_call_streaming_iterator_hook` receives the assembled response through `async_post_call_success_hook` — is pinned by a test against litellm 1.100.1, so a LiteLLM upgrade that changes it fails in CI rather than silently in a customer's billing.

### Changed
- LiteLLM proxy: the shared call id is on by default. A fresh install of `ReveniumGuardrail` with no `REVENIUM_LITELLM_SHARED_CALL_ID` set now mints the identifier, returns it as `request-id`, and reports it as the call's transaction id, so a Claude Code call that both Claude Code and the proxy report is stored once without the customer having to find the variable first. It shipped off so the first release could be observed on dev; that observation showed every call stored twice with the default and once with the id on. Set `REVENIUM_LITELLM_SHARED_CALL_ID=false` to keep LiteLLM's own response id as the transaction id and add no `request-id` header, exactly as before. The runtime guards are unchanged: on a LiteLLM without the response-header hook, under a per-tag `mode`, or under a `mode` without `pre_call` nothing is minted, and the startup warning now fires there without the variable being set; set it to `false` on a metering-only proxy to silence it.
- Claude served through Microsoft Foundry is attributed to its own `Foundry` provider bucket instead of being counted as direct Anthropic traffic. Foundry calls go through the same patched `messages` endpoints as the direct API and fell through provider detection, so Foundry spend landed in the direct-Anthropic bucket where an administrator reconciling against a Microsoft invoice could not find it. Detection is class-first from the client's MRO (so a custom gateway `base_url` cannot defeat it) with the `services.ai.azure.com` host as a fallback, and is evaluated after all Bedrock checks; `REVENIUM_BEDROCK_DISABLE` suppresses Bedrock detection only, leaving the Foundry label intact. Everything else on the payload is unchanged.
- The `litellm-proxy` extra requires `litellm[proxy]>=1.93.0`, up from `>=1.40.0`, for the `CustomGuardrail` lifecycle hooks and the deferred post-stream guardrail dispatch. The `litellm` (client) extra and `requires-python >= 3.8` are unchanged, and a client-only install pulls neither `litellm[proxy]` nor FastAPI. The guardrail module carries an import guard that raises a clear `ImportError` on Python < 3.10, which `litellm[proxy]` does not support; `revenium_middleware.litellm.proxy.ReveniumGuardrail` is `None` there and the rest of the LiteLLM integration is unaffected.
- `revenium_middleware.litellm.proxy.middleware.proxy_handler_instance` is built on first access rather than at import, so importing the SDK's LiteLLM subpackage no longer constructs the deprecated handler (and no longer emits its `DeprecationWarning`) at a caller who only uses the guardrail. The `litellm_settings.callbacks` string resolves to the same singleton as before.

### Fixed
- LiteLLM proxy: a call that reads from or writes to the prompt cache is priced once, so a gateway row costs what the provider charged. The metered `input_token_count` was LiteLLM's `prompt_tokens`, which counts the cache-read and cache-creation tokens inside it, and those same tokens were reported again in `cache_read_token_count` and `cache_creation_token_count`. Revenium prices the three separately and adds them up, so every cached token was billed at the input rate as well as at its own: one Claude Code call measured on dev recorded $0.0470 through the proxy against $0.0105 on Claude Code's own telemetry for the same call, and dashboards, budgets and alerts for that team ran on the inflated figure. For a call Anthropic served, `input_token_count` is now the tokens billed at the input rate only -- Anthropic's own `input_tokens` for the call, matching what the direct Anthropic integration reports -- with the cache counts unchanged in their own fields and the total still counting every bucket once. Every other provider's prompt count is reported exactly as before, because Revenium removes that overlap itself for the OpenAI-shaped cache pools (OpenAI, Groq, xAI, Azure, Gemini) and expects the gateway to report them gross; removing it twice would price their input at nothing. Anthropic is the one provider outside that set, which is why this is the only place its overlap can go. The upstream is read from LiteLLM's `custom_llm_provider` for the call (or the Anthropic messages route), and a call whose upstream cannot be identified keeps its prompt count untouched. Affects the `ReveniumGuardrail` post-call and streamed paths and both paths of the deprecated `MiddlewareHandler` callback. A call with no cache tokens is metered exactly as before, and rows already stored with the inflated count are not backfilled.
- LiteLLM proxy: a metered row's `request_time`, `response_time` and `completion_start_time` are the UTC instant, not the proxy machine's local wall clock. Every one of those values is published with a trailing `Z`, which means UTC, but the timestamps LiteLLM hands a logging callback carry no zone and read the machine's own clock, so a proxy outside UTC filed every row it metered from that path by its own offset. On a proxy in US Mountain time a call made at 8pm was stored as 2pm, moving evening calls onto the previous day in every hourly and daily chart and in date-range reports. That was hidden while Claude Code's own record carried the right time alongside it. With the shared call id on, which is the default, the proxy row is the one Revenium keeps, so it became the only recorded time. Affects the guardrail's streamed Anthropic route and both paths of the deprecated `MiddlewareHandler` callback. Rows metered from the non-streamed post-call path were already correct and are unchanged, as is every row on a proxy running in UTC.
- LiteLLM proxy: a **streamed** Claude Code call gets the same shared identifier as every other call on the Anthropic messages route, and the same header attribution. Streamed `/v1/messages` calls are metered from a different hook than the rest of the proxy, and that hook was added before the shared identifier and the route's header fix existed, so it filed its record under LiteLLM's own correlation id and read the request headers from the key this route does not fill. With the shared call id on, which is the default, the streamed record now carries the identifier the client was handed as `request-id`, which is what lets it merge with Claude Code's own record. Claude Code always streams, so this is the route the whole feature exists for. Your `x-revenium-*` headers are read from whichever key the route filled, matching the non-streamed record on the same route, and one call is still stored once no matter which hook reports it first. With `REVENIUM_LITELLM_SHARED_CALL_ID=false` the streamed record keeps LiteLLM's correlation id exactly as before.
- LiteLLM proxy: your `x-revenium-*` headers are read on the Anthropic messages route (`/v1/messages`), the route Claude Code and other Anthropic-shaped clients use. LiteLLM stores a request's headers under one of several keys depending on the route, and both the guardrail and the deprecated callback looked under one of them, which is not the one that route uses. The result was silent: calls were metered, but the trace id, subscriber, organization, product, subscription, task type, agent and `agenticJob*` tags you sent were all dropped, and per-subscriber budget enforcement on that route was applied to nobody. Headers are now read from whichever key the route filled, first non-empty wins, on metering and on pre-call enforcement alike. Headers set in the request body itself are read last and never override the ones LiteLLM captured, so one caller cannot attribute their spend to another.
- LiteLLM proxy: a Claude Code call made through your proxy carries its session identifier onto the metered record as the trace id, on both the guardrail and the deprecated callback. Claude Code stamps the same identifier on the usage records it reports directly, so the two views of one session can now be lined up. Note the effect on trace grouping: Revenium groups the trace list, session counts and parent-transaction joins by trace id, so a Claude Code session that previously appeared as one trace per call now appears as one trace per session. An `x-revenium-trace-id` header you set yourself always wins, and nothing about billing changes, because duplicate detection keys on the transaction id and never on the trace id.
- LiteLLM proxy: each failed attempt of a call is stored as its own record, on both the guardrail and the deprecated callback. A failure was recorded under the correlation id LiteLLM assigns the call, so two failures of one call collapsed into a single stored record, and a retried call could file its failed attempt under the same identity as the attempt you paid for. Every failure identity now ends in a marker and eight fresh characters unique to that attempt, which no successful call's identity can collide with. Correlation is unchanged: the identity still begins with the same id it did before.
- LiteLLM proxy: a failed proxied call is recorded as its own transaction. Both the guardrail and the deprecated callback metered every failure with the constant transaction id `"error-no-id"` (and the guardrail fell back to `"no-transaction-id"` on a success with no id). Revenium de-duplicates on (organization, transaction id) and passes non-UUID ids through unchanged, so for any one tenant every failed call after the first was acknowledged as a duplicate and never stored — the failure-rate signal disappeared exactly where it mattered. Each event now carries LiteLLM's own `litellm_call_id` (or the response id) when the request has one, and a fresh UUID when it does not.
- LiteLLM proxy: a failed embedding, rerank, image or transcription call is metered with its own operation type instead of being filed as `CHAT`. The success path has always read the operation off the response; the failure path hard-coded it, so the two disagreed about the same endpoint. It is now derived from the call type LiteLLM recorded, then the request endpoint, then the route the calling key was authorized against, and falls back to `CHAT` only when none of them resolves.
- LiteLLM proxy: a streamed call whose stream flag lives only on the response's own `_hidden_params` is metered with `is_streamed: true`. The payload builder re-derived the hidden params from request metadata alone rather than using the ones the hook had already resolved, so exactly the fallback branch that exists for that case was ignored.

### Deprecated
- `revenium_middleware.litellm.proxy.middleware.MiddlewareHandler` — the `litellm_settings.callbacks` entry `proxy_handler_instance` — is deprecated in favour of `ReveniumGuardrail`. It meters but never enforces a budget. It keeps working in this release and emits one `DeprecationWarning` and one log line when the proxy builds it, naming the guardrail and the README's "Migrating from the callback". Enabling both would meter every call twice; as a safety net for a proxy mid-migration, a guardrail configured to run on every request (`default_on: true` with `post_call` among its modes) claims metering ownership and the callback stops submitting rows, logging once to say so. That net deliberately does not apply to a guardrail without `default_on`, without `post_call`, or with a per-tag `Mode` that selects hooks per request — such a guardrail may not run on a given request, and suppressing the callback could drop metering entirely (a per-tag configuration logs, at info, that it is not claiming ownership). Delete the callbacks entry rather than relying on the net.

## [0.8.0] - 2026-09-10

### Fixed
- Department (org-unit) budget enforcement finds the caller whose address differs from the directory address only by letter case or surrounding whitespace. The server keys its department maps by the normalized address (trimmed, lower-cased); the SDK looked the caller up with the address exactly as the application supplied it, so those calls reached the provider and spent the money, and the block only surfaced on the next call's response.
- `BudgetExceededError.current_value` on a department budget is the caller's own balance, taken from the server's `orgUnitBudgetBlockBalances` map. For a per-person cap scoped to one department the rule's own `currentValue` is the highest single balance in that department, so the blocked developer was handed the department's top spender's figure. A payload that carries no balance map (a server predating it) still reports the rule's value, unchanged.
- A department balance the server publishes as `NaN` or `Infinity` (in either number or string form) is dropped at ingestion and at the cache lookup, so `BudgetExceededError.current_value` falls back to the rule's own `currentValue` instead of carrying a non-finite number.
- A concurrent amendment of the same job outcome is no longer a silent lost update. `report_outcome()` and `amend_outcome()` record the job's `entityVersion` from the response (readable as `JobContext.entity_version`), and the next amendment on that handle sends it as `expectedEntityVersion`, so an amendment that would overwrite a change made by another writer in between raises `OutcomeAmendConflictError` instead of winning silently. `OutcomeAmendConflictError` now carries `current_entity_version` — the version the platform actually holds, which the SDK reads off the conflict body and records on the handle — so a caller can re-check the outcome and retry with a lock instead of having nowhere to get the version from (outcome history carries an amendment sequence, not an entity version). The SDK still never auto-retries an amendment, because PATCH-amend is not idempotent. A backend that returns no version leaves every call behaving exactly as before.
- LiteLLM proxy: the callback meters the Anthropic-shaped `/v1/messages` route (pinned against litellm 1.100.0).

### Added
- An early warning before a department budget blocks you. The server publishes, with the enforcement rules, which people have crossed the warn tier of a per-person department (org-unit) cap without being blocked (`orgUnitBudgetWarnings`); the SDK now reads it and logs one warning naming the rule and — when the server publishes it — that person's own balance and the rule's threshold. Until now the first signal a developer got from the SDK was the hard block, while Slack, the webhook and the audit trail had all warned already. The line is emitted once per caller per refreshed verdict rather than once per call; a warning never blocks and never raises; a caller who is already blocked is not additionally warned; and a server that publishes no warnings map behaves exactly as before. The map is cached and persisted in the same rollback-safe, owner-only department snapshot as the block map, bound to the same rules fingerprint.
- Declared per-job metric facts on job outcomes. `JobContext.report_outcome()` and `JobContext.amend_outcome()` accept `metrics=[{"key": "quality_rate", "value": 0.93, "provenance": "MEASURED"}, ...]`, and `JobContext.append_outcome_metrics(entries)` records facts that only become measurable after the outcome was reported (`AgenticOutcomeClient.append_outcome_metrics()` is the low-level equivalent). Entries are forwarded exactly as supplied, so `provenance`, `recordedBy` and `source` keep their documented server-side defaults instead of being misattributed client-side; the SDK shape-checks each entry (a non-blank `key`, a `value`) before any request and leaves the economics rules to the platform, which requires the metric to be declared `PER_JOB` on the job type and range-checks `quality_rate` to 0..1. Facts are what AI Alerts evaluate `QUALITY_RATE` from, so an integration that could not emit them was invisible to those rules. Appending facts does not advance the job's `entityVersion`, so a handle keeps the version it recorded and a following `amend_outcome()` still locks against it. The append is retried only on `429` and never on an ambiguous gateway error, because a repeated append records a second fact rather than the same one.
- Agent version attribution on all four AI metering endpoints. Set `usage_metadata={"agent_version": "1.4.2"}` (or `agentVersion`) to record which release of your agent produced a call, and cost breaks down per agent version. Accepted on completions, audio, image and video together, forwarded by every provider integration, and capped at 64 characters to match the ingest contract. This is the agent's own version and is distinct from `agentic_job_version`, which versions the agentic job definition; the two never overwrite each other.
- Job type economics: `upsert_job_type_economics`, `get_job_type_economics`, `report_period_facts`, `create_baseline` and `list_baselines` (with the `JobTypeEconomics`, `PeriodFactEntry` and `Baseline` models) declare a job type's metrics and economics, append period facts and versioned baselines. Appends (facts, baselines) are retried only on `429`, never on an ambiguous gateway error, because a repeated append records a second fact.

### Changed
- `JobContext.amend_outcome()` accepts `expected_entity_version=` (the optimistic lock, defaulting to the version recorded by the handle's last report or amendment and omitted when none is known), and `reason` is now optional: an API-key caller may omit it and the platform records an automated correction reason derived from the source. `reason` remains the first positional argument, and a blank string is still rejected client-side. A version that is not a non-negative integer (`3.5`, `-1`, `True`) is rejected before the request rather than truncated.
- `AgenticOutcomeClient.create_job()` returns the created job resource parsed from the response — including its `entityVersion` — merged over the request body it sent, so a response that omits a field the caller supplied never drops it. An idempotent 409 re-run, a dry run and a bodiless response return the request body alone.
- Agentic outcomes documentation now standardizes on `REVENIUM_WRITE_API_KEY` as the primary write-scope key name. `REVENIUM_OUTCOME_API_KEY` remains documented as a deprecated fallback, and the documented resolution order is explicit `api_key=`, then `REVENIUM_WRITE_API_KEY`, then `REVENIUM_OUTCOME_API_KEY`, then `REVENIUM_METERING_API_KEY`.
- Enforcement rule refresh honours the server's `Retry-After` in full up to 20 seconds per call and then gives up for that cycle instead of refetching; the refresh cooldown is capped at 300 seconds so one absurd header cannot freeze rule refreshes.

## [0.7.0] - 2026-08-28

### Added
- Reasoning effort on completion metering. Record the effort level requested of the model (`usage_metadata={"effort": ...}` or `REVENIUM_EFFORT`); unset effort is omitted from the wire, never sent as null.
- `model_host` and `subscriber_email_source` on AI completion metering — where the model actually ran (bedrock, vertex, anthropic-api) and how the subscriber email was obtained, matching what OTLP ingestion already carries.
- Tool usage events join their agentic job: `@meter_tool` / `report_tool_call` resolve `agentic_job_id` with the same precedence as completions (call metadata, then JobContext, then environment). An id the server would reject is dropped with a debug log so the event and its cost still deliver — losing attribution, never the event.
- Department (org-unit) budget enforcement in the pre-call circuit breaker. The poller now consumes the server-computed department block map and blocks exactly the callers the platform says are over budget; the on-disk snapshot is rollback-safe (rules keep the legacy shape older SDKs read), the department map is owner-only (0600) because it is keyed by email, and it is content-bound to its rules snapshot so a torn write can never pair it with the wrong rules.
- Per-group balances honored when evaluating subscriber-grouped enforcement rules.
- Service-tier and pricing fields, billing-skip/error/cache-TTL telemetry fields on completion metering; audio, image and video payloads aligned with the current platform spec.
- Anthropic per-TTL cache-creation buckets forwarded from the native middleware.
- `outcomeReason` across job outcome reporting, amendment and history — automatic FAILED reports populate it from the exception (truncated to the ingest cap).

### Fixed
- OpenAI reasoning token counts are read from the response instead of hardcoded to zero.
- LiteLLM proxy middleware populates cache token counts, on success and failure paths alike.
- Flat subscriber attribution fields are matched when nested metadata suppresses them; the nested subscriber email is authoritative when both are present; group matching fails open on non-string subscriber identifiers.

## [0.6.0] - 2026-08-13

### Added
- Agentic job and squad attribution as first-class typed parameters. `agentic_job_id`, `agentic_job_name`, `agentic_job_type`, `agentic_job_version`, `squad_id`, `squad_name`, and `squad_role` are now typed, documented parameters on all four AI metering endpoints (completions, audio, image, video) instead of requiring the `extra_body` escape hatch. Existing `extra_body` callers keep working unchanged.
- The media metering endpoints (audio, image, video) now accept `operation_type` and the prompt-capture fields (`input_messages`, `output_response`, `prompts_truncated`), matching the completion endpoint.
- Ticket attribution on media metering events. The `ticket_id` trace field (set per call via `usage_metadata={"ticket_id": ...}` or process-wide via `REVENIUM_TICKET_ID`) is now sent on audio, image, and video metering events, matching the existing completion behavior. Previously a configured ticket ID was silently dropped on every media event; fal.ai and Google media calls now carry it end to end.
- Skill attribution on completion metering events. Six new optional fields — `skill_name`, `skill_kind`, `skill_source`, `skill_plugin_name`, `skill_marketplace_name`, and `skill_invocation_trigger` — attribute AI usage to the skill that produced it. Set them per call via `usage_metadata` (snake_case or camelCase) or process-wide via the `REVENIUM_SKILL_*` environment variables; each field resolves independently and unset fields are omitted. Wired into the OpenAI middleware; other providers will follow once the field semantics are confirmed against a released backend.
- Metering error visibility. Metering is fire-and-forget by design — a delivery failure never interrupts your AI calls — which previously meant failures were only visible as easily-missed log lines. Two new mechanisms make them observable: `revenium_middleware.on_metering_error(callback)` invokes your callback with the exception, the operation type, and a timestamp every time a metering event fails to deliver, and `revenium_middleware.get_metering_status()` returns running success/error counters plus the most recent error. Callbacks run on the background metering thread and can never raise into your application.

### Changed
- Metering delivery failures (including HTTP 4xx/5xx responses from Revenium) are now logged at ERROR level instead of WARNING, across all providers and for tool events.
- A missing `REVENIUM_METERING_API_KEY` now logs a clear ERROR at initialization stating that metering is disabled, instead of a quiet warning.
- When a provider's SDK is installed but the Revenium middleware for it fails to import (for example a missing or conflicting dependency), the failure is now logged at ERROR level with the underlying import error, instead of a DEBUG line that hid the problem. Providers whose SDK is not installed continue to be skipped quietly.

## [0.5.0] - 2026-08-05

### Added
- Griptape framework integration, available as the `griptape` extra. Install with `pip install "revenium-python-sdk[griptape,<provider>]"` and import drivers from `revenium_middleware.griptape`. The universal `ReveniumDriver` auto-detects the provider from the model name (OpenAI, Anthropic, Ollama, or anything else via LiteLLM) and wraps the matching Griptape prompt driver with metering; `ReveniumEmbeddingDriver` does the same for embeddings, and provider-specific drivers are available for direct control. This replaces the standalone `revenium-griptape` package; all driver class names are unchanged. Requires Python 3.10+.

### Changed
- Every optional extra is now installed in a clean, isolated environment as part of the test suite run on each change, so a dependency that stops resolving is caught before release.

## [0.4.0] - 2026-07-29

### Added
- Agentic job tracking with `JobContext`. Wrapping a unit of agent work in `with JobContext("loan-app-12345", type="loan_processing") as job:` meters every AI call made inside the block against that job — across all supported providers, with no per-call changes — and `job.report_outcome(...)` records the business result (`SUCCESS`, `FAILED`, or `CANCELLED`, plus an outcome type, value, and currency) so dashboards can show revenue impact next to AI spend. `async with` behaves identically. If an unhandled exception escapes the block before an outcome was reported, the job is automatically recorded as `FAILED` with the error details in metadata and the original exception is re-raised untouched. Nested contexts are treated as separate jobs, and exiting an inner context restores the outer job's fields.
- Outcome amendments and history. Business results change after the fact, so recorded outcomes can now be corrected instead of duplicated: `JobContext.attach("sales-lead-8842")` returns a lightweight handle to an existing job — usable from a different process than the one that ran it — whose `amend_outcome(reason=..., ...)` updates the stored outcome, while `get_outcome_history()` returns the full amendment trail in order. A dedicated exception family makes each failure mode explicit and catchable through the shared `OutcomeReportingError` base: `OutcomeAlreadyReportedError` (job already has an outcome — amend instead), `OutcomeNotReportedError` (nothing to amend yet), and `OutcomeAmendConflictError` (a concurrent amendment won the optimistic lock).
- Agentic job fields without a context manager, for code that cannot wrap its work in a block: the `@track_job()` decorator for LiteLLM (static values or extraction from function arguments), per-call `usage_metadata={"agentic_job_id": ...}`, or the process-wide `REVENIUM_AGENTIC_JOB_ID`, `REVENIUM_AGENTIC_JOB_NAME`, `REVENIUM_AGENTIC_JOB_TYPE`, and `REVENIUM_AGENTIC_JOB_VERSION` environment variables.
- Job outcome reporting for CrewAI. `ReveniumCrewWrapper` accepts `agentic_job_id`, `agentic_job_name`, `agentic_job_type`, and `agentic_job_version`, tying every LLM call the crew makes to a single job, and exposes `report_job_outcome()` and `amend_job_outcome()` for the business result.
- `ticket_id` trace field, threaded through metering for every provider integration (OpenAI, Anthropic including AWS Bedrock, Google Gemini and Vertex AI, Ollama, LiteLLM, Perplexity, and fal.ai). Set it per call via `usage_metadata={"ticket_id": "JIRA-123"}` or process-wide via `REVENIUM_TICKET_ID` to attribute AI cost to an individual ticket or issue.
- Tunable blocking behavior for outcome calls. Outcome reporting, amendment, and history retrieval are synchronous HTTP requests with retries; `retry_attempts`, `retry_initial_seconds`, and `retry_max_seconds` bound how long they may block, and are accepted by the `JobContext` constructor, `JobContext.attach()`, `get_outcome_history()`, and the CrewAI wrapper's outcome helpers.

### Changed
- The agentic outcomes API (reporting, amending, and reading job outcomes) requires a write-scope API key (`rev_sk_`). Metering keys (`rev_mk_`) meter completions and tool events only; the SDK now rejects them client-side with a clear error before any HTTP request is made, instead of surfacing a server-side authorization failure. Key resolution order is explicit `api_key=`, then `REVENIUM_OUTCOME_API_KEY`, then `REVENIUM_METERING_API_KEY`. The outcomes API base URL is configurable via `REVENIUM_PROFITSTREAM_BASE_URL`.
- Team resolution for outcome reporting follows explicit `team_id=`, then `REVENIUM_TEAM_ID`, then automatic resolution from the API key, raising `OutcomeReportingError` if none of those yields a team.

## [0.3.0] - 2026-07-17

### Added
- Opt-in transport-level metering for AWS Bedrock (canary). Setting `REVENIUM_BEDROCK_TRANSPORT=1` meters raw `boto3.client("bedrock-runtime")` usage — `invoke_model`, `invoke_model_with_response_stream`, `converse`, and `converse_stream` — for Anthropic Claude models with no application code changes. Each physical invocation produces exactly one completion attributed to AWS, streaming calls report time-to-first-token from the first event, and the AWS response request ID is used as the transaction identity so provider retries can never double-bill. The same variable acts as a runtime kill switch (it is re-read on every call). Calls made internally by the SDK's own Bedrock integration are automatically excluded from transport metering, so nothing is double-counted. Disabled by default; requires boto3.

### Fixed
- Async Anthropic clients routed through AWS Bedrock (`AsyncAnthropicBedrock`) are now attributed to AWS in usage reports instead of direct Anthropic.
- Fully qualified Bedrock model IDs and inference-profile identifiers (e.g. `us.anthropic.claude-...`, ARNs) are now preserved byte-for-byte instead of being double-prefixed into invalid IDs. Known short aliases keep resolving as before, and bare model names still receive the `anthropic.` prefix.
- LiteLLM: cached prompt tokens are now read from the correct field of the response and reported as cache reads. Previously the count was always zero and, when present, would have been misclassified as cache writes. Anthropic-style cache fields surfaced through LiteLLM are also propagated.
- Google Gemini and Vertex AI: `cachedContentTokenCount` is now reported as cache reads instead of cache creation, so cache-heavy Gemini workloads are costed correctly.
- README code samples now use the package's real import paths (`import revenium_middleware.openai` and friends). The previously documented underscore module names belong to retired standalone packages and fail under this bundle. The LangChain section was rewritten around the bundled API (`wrap`/`attach_to`).

## [0.2.0] - 2026-07-09

### Added
- Store-and-forward buffering for metering events. Events that fail delivery with a transient error (connection failures, timeouts, rate limiting, or server errors) are no longer dropped: they are held in a bounded in-memory buffer and automatically replayed in the background once the backend is reachable again. Replayed events keep their original `Idempotency-Key`, so the backend safely deduplicates any overlap. The buffer is tunable via `REVENIUM_BUFFER_MAX_SIZE` (default 1000 events, oldest evicted first) and `REVENIUM_BUFFER_FLUSH_INTERVAL` (default 30 seconds); buffered events expire after 24 hours. A new `get_buffer_stats()` helper exposes buffer counters for observability, and a best-effort final drain runs at process exit.
- `initialize_metering(api_key=..., base_url=...)` for explicit programmatic (re)configuration of the metering client at runtime — useful for credential rotation or for configuring metering after import. Invalid keys raise immediately instead of failing silently.

### Changed
- Enforcement rule fetches now ride out transient failures (rate limits, server errors, connection errors) with exponential backoff capped at 8 seconds, honoring `Retry-After` headers. If retries are exhausted the SDK fails open and keeps the previously cached rules — an enforcement-refresh outage never blocks application traffic.
- Tool metering (`meter_tool` / `report_tool_call`) now dispatches events fire-and-forget on a background thread. Decorated functions and manual reporting return immediately instead of blocking on the metering network round-trip.

### Fixed
- Anthropic: raw streaming via `client.messages.create(stream=True)` is now metered. Previously only the `client.messages.stream()` context-manager path produced metering events.
- Streams abandoned before completion (early `break`, explicit `close()`, or garbage collection) now submit a metering event with the usage observed up to that point instead of silently dropping it. Applies to sync and async streams across providers.
- OpenAI streaming: when the middleware injects `stream_options={"include_usage": true}` to obtain token usage, the synthetic final usage chunk is no longer surfaced to callers that did not request it — the visible chunk sequence now matches the raw SDK exactly.
- Metering credentials set via environment variables after `revenium_middleware` is imported are now picked up on the next metering call; there is no longer an import-order requirement.
- Tool metering honors `configure()` overrides and environment variables at dispatch time and uses the correct metering API path.
- Anthropic on AWS Bedrock: the adapter now resolves the AWS region following boto3 conventions (client configuration, then `AWS_REGION`, then `AWS_DEFAULT_REGION`) instead of always defaulting to `us-east-1`.
- Anthropic on AWS Bedrock: genuine API errors from Bedrock calls now propagate to the caller instead of being masked by an internal fallback path.
- Streaming metering now reports `completion_start_time` as the arrival of the first stream event, improving time-to-first-token accuracy.
- DEBUG-level logging no longer leaks prompt or message content and credentials: sensitive fields (API keys, authorization headers, tokens) are redacted and message content is summarized across provider integrations.

## [0.1.10] - 2026-07-03

### Changed
- The SDK is now fully self-contained. The metering client that was previously installed as a separate `revenium-metering` package now ships inside `revenium-python-sdk`, so `pip install revenium-python-sdk` is all you need — no second Revenium package is pulled in. Public imports are unchanged: `from revenium_middleware import meter_tool, report_tool_call, configure` keeps working exactly as before.

## [0.1.9] - 2026-06-18

### Changed
- The agentic-outcomes example pack now produces more realistic output: per-job variability in outcomes and costs, corrected success/unsuccessful counts in the run summary, and jittered escalation costs. Adds a `load-demo.sh` helper for running the demo pack, plus README clarifications.

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

[0.9.0]: https://github.com/revenium/revenium-python-sdk/releases/tag/v0.9.0
[0.8.0]: https://github.com/revenium/revenium-python-sdk/releases/tag/v0.8.0
