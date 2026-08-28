"""Agentic-outcome client used by the demo example pack.

Public surface:
- AgenticOutcomeSettings: immutable config (single API key authenticates every call).
- AgenticOutcomeClient: emit_completion / emit_tool_event / report_outcome.

Single-key contract: settings.api_key authenticates both metering (via the
in-package ReveniumMetering client) and the raw-HTTP fallback for endpoints the SDK does
not yet cover (/meter/v2/tool/events, /profitstream/v2/api/jobs/{id}/outcome).
"""

from __future__ import annotations

import json
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from revenium_middleware._core import outcomes as _outcomes
from revenium_middleware._metering import ReveniumMetering


@dataclass(frozen=True)
class AgenticOutcomeSettings:
    api_key: str
    meter_base_url: str = "https://api.revenium.io"
    profitstream_base_url: str = "https://api.revenium.io"
    team_id: str = ""
    # When set, used to authenticate /profitstream/v2/api/jobs[/outcome] calls.
    # Falls back to api_key. Lets callers use a metering key (rev_mk_) for ingestion
    # and a write key (rev_sk_) for job/outcome control.
    outcome_api_key: Optional[str] = None
    outcome_retry_attempts: int = 10
    outcome_retry_initial_seconds: float = 2.0
    # Hard cap on any single retry sleep (also clamps server-supplied Retry-After).
    outcome_retry_max_seconds: float = 90.0

    @property
    def tool_url(self) -> str:
        return f"{self.meter_base_url.rstrip('/')}/meter/v2/tool/events"

    @property
    def completion_url(self) -> str:
        return f"{self.meter_base_url.rstrip('/')}/meter/v2/ai/completions"

    @property
    def sdk_meter_base_url(self) -> str:
        root = self.meter_base_url.rstrip("/")
        if root.endswith("/meter"):
            return f"{root}/"
        return f"{root}/meter/"


class AgenticOutcomeClient:
    """Emit AI completions, tool events, and terminal job outcomes."""

    def __init__(
        self,
        settings: AgenticOutcomeSettings,
        *,
        metering_client: Optional[ReveniumMetering] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.settings = settings
        self._metering_client = metering_client
        self._metering_lock = threading.Lock()
        self.http_client = http_client or httpx.Client(timeout=60.0)
        self._owns_http_client = http_client is None
        self._resolved_team_id: Optional[str] = None

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def __enter__(self) -> "AgenticOutcomeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _metering(self) -> ReveniumMetering:
        if self._metering_client is None:
            with self._metering_lock:
                if self._metering_client is None:
                    self._metering_client = ReveniumMetering(
                        api_key=self.settings.api_key,
                        base_url=self.settings.sdk_meter_base_url,
                    )
        return self._metering_client

    # ------------------------------------------------------------------ completions

    def emit_completion(self, payload: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
        if dry_run:
            _print_dry_run("POST", self.settings.completion_url, payload)
            return payload

        # Mandatory keys in the payload to prevent KeyError
        required = [
            "completionStartTime",
            "costType",
            "inputTokenCount",
            "isStreamed",
            "model",
            "outputTokenCount",
            "provider",
            "requestDuration",
            "requestTime",
            "responseTime",
            "stopReason",
            "totalTokenCount",
            "transactionId",
        ]
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"Missing required keys in payload: {', '.join(missing)}")

        extra_body = {
            key: payload[key]
            for key in (
                "agentName",
                "agenticJobId",
                "agenticJobName",
                "agenticJobType",
                "agenticJobVersion",
                "squadName",
                "squadId",
                "currency",
            )
            if key in payload
        }
        self._metering().ai.create_completion(
            completion_start_time=payload["completionStartTime"],
            cost_type=payload["costType"],
            input_token_count=payload["inputTokenCount"],
            is_streamed=payload["isStreamed"],
            model=payload["model"],
            output_token_count=payload["outputTokenCount"],
            provider=payload["provider"],
            request_duration=payload["requestDuration"],
            request_time=payload["requestTime"],
            response_time=payload["responseTime"],
            stop_reason=payload["stopReason"],
            total_token_count=payload["totalTokenCount"],
            transaction_id=payload["transactionId"],
            agent=payload.get("agent"),
            cache_creation_token_count=payload.get("cacheCreationTokenCount"),
            cache_read_token_count=payload.get("cacheReadTokenCount"),
            input_token_cost=payload.get("inputTokenCost"),
            model_source=payload.get("modelSource"),
            operation_type=payload.get("operationType"),
            organization_name=payload.get("organizationName"),
            output_token_cost=payload.get("outputTokenCost"),
            product_name=payload.get("productName"),
            reasoning_token_count=payload.get("reasoningTokenCount"),
            # Only pass the reasoning effort level when the caller supplied
            # one: create_completion drops NotGiven but keeps an explicit
            # None, which would reach the wire as "effort": null. A key
            # present with None counts as not supplied — callers forwarding
            # their own .get("effort") must not reintroduce the null.
            **({"effort": payload["effort"]} if payload.get("effort") is not None else {}),
            subscriber=payload.get("subscriber"),
            task_type=payload.get("taskType"),
            time_to_first_token=payload.get("timeToFirstToken"),
            total_cost=payload.get("totalCost"),
            trace_id=payload.get("traceId"),
            environment=payload.get("environment"),
            parent_transaction_id=payload.get("parentTransactionId"),
            retry_number=payload.get("retryNumber"),
            trace_name=payload.get("traceName"),
            trace_type=payload.get("traceType"),
            transaction_name=payload.get("transactionName"),
            system_prompt=payload.get("systemPrompt"),
            input_messages=payload.get("inputMessages"),
            output_response=payload.get("outputResponse"),
            prompts_truncated=payload.get("promptsTruncated"),
            error_reason=payload.get("errorReason"),
            extra_body=extra_body or None,
            # Per-instance client; idempotency key generated inline (see FRONT-1208 spec exception).
            extra_headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        return payload

    # ------------------------------------------------------------------ tool events

    def emit_tool_event(self, payload: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
        if dry_run:
            _print_dry_run("POST", self.settings.tool_url, payload)
            return payload
        _outcomes.post_with_retry(
            self.http_client,
            self.settings.tool_url,
            params=None,
            body=payload,
            api_key=self.settings.api_key,
            retry_attempts=self.settings.outcome_retry_attempts,
            retry_initial_seconds=self.settings.outcome_retry_initial_seconds,
            retry_max_seconds=self.settings.outcome_retry_max_seconds,
        )
        return payload

    # ------------------------------------------------------------------ jobs+outcomes

    def create_job(
        self,
        agentic_job_id: str,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
        version: Optional[str] = None,
        environment: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.settings.profitstream_base_url.rstrip('/')}/profitstream/v2/api/jobs"
        body: Dict[str, Any] = {"agenticJobId": agentic_job_id}
        if name:
            body["name"] = name
        if type:
            body["type"] = type
        if version:
            body["version"] = version
        if environment:
            body["environment"] = environment
        if dry_run:
            _print_dry_run("POST", url, body)
            return body
        team_id = self._get_team_id()
        params = {"teamId": team_id} if team_id else None
        # 409 Conflict is acceptable (idempotent re-runs of the same agenticJobId).
        _outcomes.post_with_retry(
            self.http_client,
            url,
            params=params,
            body=body,
            api_key=self._outcome_key(),
            accept_409=True,
            retry_attempts=self.settings.outcome_retry_attempts,
            retry_initial_seconds=self.settings.outcome_retry_initial_seconds,
            retry_max_seconds=self.settings.outcome_retry_max_seconds,
        )
        return body

    def report_outcome(
        self, agentic_job_id: str, payload: Dict[str, Any], *, dry_run: bool = False
    ) -> None:
        """Report the terminal outcome for a job.

        The payload is sent as-is, so any documented outcome field can be
        included, e.g.::

            {"executionStatus": "FAILED", "outcomeType": "UNSUCCESSFUL",
             "outcomeReason": "Customer unreachable after three attempts"}

        outcomeReason is the prescribed field for a failure explanation — do not
        encode it inside metadata. It is read-only on the job resource itself,
        which is why create_job never sends it.

        Raises OutcomeAlreadyReportedError when the job already has an outcome
        (backend 409 with structured amendment guidance).
        """
        safe_id = urllib.parse.quote(agentic_job_id, safe="")
        url = (
            f"{self.settings.profitstream_base_url.rstrip('/')}"
            f"/profitstream/v2/api/jobs/{safe_id}/outcome"
        )
        if dry_run:
            _print_dry_run("POST", url, payload)
            return
        team_id = self._get_team_id()
        params = {"teamId": team_id} if team_id else None
        # 404 transient: metric ingestion creates the Job asynchronously; a fresh
        # outcome can race ahead until Kafka catches up.
        _outcomes.post_with_retry(
            self.http_client,
            url,
            params=params,
            body=payload,
            api_key=self._outcome_key(),
            retry_on_404=True,
            raise_typed_on_409=True,
            retry_attempts=self.settings.outcome_retry_attempts,
            retry_initial_seconds=self.settings.outcome_retry_initial_seconds,
            retry_max_seconds=self.settings.outcome_retry_max_seconds,
        )

    def _outcome_key(self) -> str:
        # Job/outcome control requires a write-scope key (rev_sk_); a metering
        # key (rev_mk_) is rejected here so callers fail fast client-side
        # instead of on a backend 403. Metering paths keep using api_key as-is.
        return _outcomes.validate_outcome_key(
            self.settings.outcome_api_key or self.settings.api_key
        )

    def _get_team_id(self) -> str:
        if self.settings.team_id:
            return self.settings.team_id
        if self._resolved_team_id:
            return self._resolved_team_id

        # Delegate auto-resolution (key-prefix + ProfitStream teams API) to the
        # shared transport. use_env=False preserves this client's historical
        # chain: settings.team_id > cached auto-resolution — REVENIUM_TEAM_ID
        # is already folded into settings by the examples-pack loader.
        resolved = _outcomes.resolve_team_id(
            "",
            self._outcome_key(),
            self.http_client,
            self.settings.profitstream_base_url,
            use_env=False,
        )
        if resolved:
            self._resolved_team_id = resolved
        return resolved


def _print_dry_run(method: str, url: str, payload: Dict[str, Any]) -> None:
    print(f"[DRY-RUN] {method} {url}")
    print(json.dumps(payload, indent=2))
