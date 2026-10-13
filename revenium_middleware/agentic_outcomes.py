"""Agentic-outcome client used by the demo example pack.

Public surface:
- AgenticOutcomeSettings: immutable config (single API key authenticates every call).
- AgenticOutcomeClient: emit_completion / emit_tool_event / report_outcome.

Single-key contract: settings.api_key authenticates both metering (via the
in-package ReveniumMetering client) and the raw-HTTP fallback for endpoints not provided
by that client (/meter/v2/tool/events, /profitstream/v2/api/jobs/{id}/outcome).

emit_completion and emit_tool_event queue their record and return at once; it
is delivered like every automatic record (BACK-4451).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, Coroutine, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import httpx
from revenium_middleware._core import outcomes as _outcomes
from revenium_middleware._core.config import (
    Config,
    get_profitstream_base_url,
    get_revenium_address,
    revenium_address_from_usage_address,
)
from revenium_middleware._core.context import snapshot_metadata
from revenium_middleware._core.metering import (
    _scoped_client_for,
    current_client,
    recorded_under,
    run_async_in_thread,
)
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._core.trace_fields import validate_agent_version
from revenium_middleware._metering import ReveniumMetering
from revenium_middleware._metering._exceptions import APIError
from revenium_middleware._metering.decorator import deliver_tool_event
from revenium_middleware._metering._utils._transform import _maybe_transform_key, get_type_hints
from revenium_middleware._metering.types.ai_create_completion_params import AICreateCompletionParams

logger = logging.getLogger(__name__)

_REQUIRED_COMPLETION_KEYS = (
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
)

_EXTRA_BODY_COMPLETION_KEYS = (
    "agenticJobId",
    "agenticJobName",
    "agenticJobType",
    "agenticJobVersion",
    "squadName",
    "squadId",
)

# Read off the metering client's own parameter type, so a field added there is
# forwarded by emit_completion without an edit here.
_COMPLETION_KEYWORD_BY_WIRE_NAME: Dict[str, str] = {
    _maybe_transform_key(keyword, hint): keyword
    for keyword, hint in get_type_hints(AICreateCompletionParams, include_extras=True).items()
}


def _sdk_meter_base_url(address: str) -> str:
    root = address.rstrip("/")
    if root.endswith("/meter"):
        return f"{root}/"
    return f"{root}/meter/"


def _tool_url(address: str) -> str:
    return f"{address.rstrip('/')}/meter/v2/tool/events"


def _completion_url(address: str) -> str:
    return f"{address.rstrip('/')}/meter/v2/ai/completions"


def _outcome_address_is_configured() -> bool:
    configured = os.getenv(Config.ENV_REVENIUM_PROFITSTREAM_BASE_URL)
    return bool(configured and configured.strip())


@dataclass(frozen=True)
class AgenticOutcomeSettings:
    """Immutable settings for ``AgenticOutcomeClient``.

    ``meter_base_url`` defaults to the usage address in
    ``REVENIUM_METERING_BASE_URL`` (without its ``/meter`` segment), else
    ``https://api.revenium.ai``. ``profitstream_base_url`` defaults to
    ``REVENIUM_PROFITSTREAM_BASE_URL`` when that is set, else it follows
    ``meter_base_url``, so one address serves usage and outcomes unless a
    separate outcome address is given on purpose. Both are resolved when the
    settings are built: ``dataclasses.replace(settings, meter_base_url=...)``
    keeps the outcome address already resolved.

    While ``meter_base_url`` equals the environment's address (it was not
    passed, or was passed with that same value), usage records follow the
    address set in code with ``initialize_metering(base_url=...)`` or a
    ``metering_key(..., base_url=...)`` block, as automatic records do. Jobs and
    outcomes follow it too, unless ``profitstream_base_url`` names another
    address or ``REVENIUM_PROFITSTREAM_BASE_URL`` is set, so a job and its
    outcome never land on two different addresses.
    """

    api_key: str
    meter_base_url: str = field(default_factory=get_revenium_address)
    profitstream_base_url: str = ""
    team_id: str = ""
    # When set, used to authenticate /profitstream/v2/api/jobs[/outcome] calls.
    # Falls back to api_key. Lets callers use a metering key (rev_mk_) for ingestion
    # and a write key (rev_sk_) for job/outcome control.
    outcome_api_key: Optional[str] = None
    outcome_retry_attempts: int = 10
    outcome_retry_initial_seconds: float = 2.0
    # Hard cap on any single retry sleep (also clamps server-supplied Retry-After).
    outcome_retry_max_seconds: float = 90.0
    _follows_sdk_address: bool = field(default=False, init=False, repr=False, compare=False)
    _outcomes_follow_usage_address: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # The dataclass is frozen, so derived and trimmed values are set the
        # way dataclasses itself sets fields.
        for key_field in ("api_key", "outcome_api_key"):
            value = getattr(self, key_field)
            if value is not None:
                object.__setattr__(self, key_field, value.strip())
        if not self.profitstream_base_url:
            object.__setattr__(
                self,
                "profitstream_base_url",
                get_profitstream_base_url(metering_base_url=self.meter_base_url),
            )
        follows_sdk_address = self.meter_base_url == get_revenium_address()
        object.__setattr__(self, "_follows_sdk_address", follows_sdk_address)
        object.__setattr__(
            self,
            "_outcomes_follow_usage_address",
            follows_sdk_address
            and not _outcome_address_is_configured()
            and self.profitstream_base_url == get_profitstream_base_url(metering_base_url=self.meter_base_url),
        )

    @property
    def tool_url(self) -> str:
        return _tool_url(self.meter_base_url)

    @property
    def completion_url(self) -> str:
        return _completion_url(self.meter_base_url)

    @property
    def sdk_meter_base_url(self) -> str:
        return _sdk_meter_base_url(self.meter_base_url)


class AgenticOutcomeClient:
    """Emit AI completions, tool events, and terminal job outcomes.

    ``emit_completion`` and ``emit_tool_event`` send through the SDK's
    background delivery, with its timeouts, retries and outage buffer. A
    ``metering_client`` passed here sends the completions; ``http_client``
    serves jobs and outcomes only.
    """

    def __init__(
        self,
        settings: AgenticOutcomeSettings,
        *,
        metering_client: Optional[ReveniumMetering] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.settings = settings
        self._metering_client = metering_client
        self.http_client = http_client or httpx.Client(timeout=60.0)
        self._owns_http_client = http_client is None
        self._resolved_team_ids: Dict[str, str] = {}
        self._reported_unknown_keys: Set[str] = set()

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def __enter__(self) -> "AgenticOutcomeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _usage_address(self) -> str:
        """Where this call's usage record goes: the SDK's current address when the settings follow it."""
        if self.settings._follows_sdk_address:
            sdk_client = current_client()
            if sdk_client is not None:
                return revenium_address_from_usage_address(str(sdk_client.base_url))
        return self.settings.meter_base_url

    def _outcome_address(self) -> str:
        if self.settings._outcomes_follow_usage_address:
            return get_profitstream_base_url(metering_base_url=self._usage_address())
        return self.settings.profitstream_base_url

    def _metering(self) -> ReveniumMetering:
        if self._metering_client is not None:
            return self._metering_client
        return _scoped_client_for(self.settings.api_key, _sdk_meter_base_url(self._usage_address()))

    # ------------------------------------------------------------------ completions

    def emit_completion(self, payload: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
        """Meter one LLM call from a payload keyed by the completions API's field names.

        Every field the metering client accepts is forwarded. An optional field
        that is absent or ``None`` is left out of the request; a required field
        is sent as given. A key the API does not know is not sent, and is named
        in a warning once per client.

        The record is queued and sent in the background, like automatic
        records: delivery errors are logged, not raised, and a retryable one
        leaves the record in the outage buffer for replay. Returns ``payload``.

        Raises:
            ValueError: a required key is missing; nothing is queued.
        """
        if dry_run:
            _print_dry_run("POST", _completion_url(self._usage_address()), payload)
            return payload

        missing = [k for k in _REQUIRED_COMPLETION_KEYS if k not in payload]
        if missing:
            raise ValueError(f"Missing required keys in payload: {', '.join(missing)}")

        keywords, extra_body, unknown_keys = _split_completion_payload(payload)
        self._warn_once_about_unknown_keys(unknown_keys)
        args = snapshot_metadata({**keywords, "extra_body": extra_body or None})
        # One key per call, frozen now, so a buffer replay is deduplicated.
        idempotency_key = str(uuid.uuid4())
        transaction_id = payload.get("transactionId")
        with recorded_under(self._metering()):
            _queue(_send_completion(args, idempotency_key, transaction_id))
        return payload

    def _warn_once_about_unknown_keys(self, unknown_keys: List[str]) -> None:
        new_keys = set(unknown_keys) - self._reported_unknown_keys
        if not new_keys:
            return
        self._reported_unknown_keys |= new_keys
        logger.warning(
            "emit_completion did not send payload keys the completions API does not accept: %s",
            ", ".join(sorted(new_keys)),
        )

    # ------------------------------------------------------------------ tool events

    def emit_tool_event(self, payload: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
        """Meter one tool call; queued and sent in the background, like automatic tool events.

        One attempt with the metering timeouts; a retryable failure leaves the
        event in the outage buffer for replay. Delivery errors are logged, not
        raised. Returns ``payload``.
        """
        tool_url = _tool_url(self._usage_address())
        if dry_run:
            _print_dry_run("POST", tool_url, payload)
            return payload
        tool_event = {
            "url": tool_url,
            "key": self.settings.api_key,
            "event_payload": snapshot_metadata(payload),
            # One key for the first send and every replay of this call: the
            # tool events API replays its first answer for a repeated key, so
            # a replay after a gateway error is not stored twice. Not the
            # payload's transactionId, which a caller may reuse with a
            # different body and which the API then refuses as a key mismatch.
            "idempotency_key": str(uuid.uuid4()),
        }
        transaction_id = payload.get("transactionId")
        _queue(_send_tool_event(tool_event, transaction_id), gated_by_circuit=False)
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
        """Create (or re-affirm) a job. Returns the created job resource.

        The response carries the job's ``entityVersion``, which the outcome
        PATCH accepts back as ``expectedEntityVersion`` — returning the parsed
        body lets a caller seed that version without a second read of the job.

        The response is merged over the request body rather than replacing it,
        and the request body is returned alone when the response cannot supply
        one — a 409 idempotent re-run answers with an error body rather than a
        job resource, and a dry run or a bodiless 2xx has nothing to parse. So
        the return value always still carries every field the caller supplied.
        """
        url = f"{self._outcome_address().rstrip('/')}/profitstream/v2/api/jobs"
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
        response = _outcomes.post_with_retry(
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
        return _job_resource_or(body, response)

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
        (backend 409 with structured amendment guidance). A report that was
        stored but answered with a gateway error is recognised on the retry and
        returns normally; see ``report_outcome_request``.
        """
        outcome_address = self._outcome_address()
        safe_id = urllib.parse.quote(agentic_job_id, safe="")
        url = (
            f"{outcome_address.rstrip('/')}"
            f"/profitstream/v2/api/jobs/{safe_id}/outcome"
        )
        if dry_run:
            _print_dry_run("POST", url, payload)
            return
        team_id = self._get_team_id()
        _outcomes.report_outcome_request(
            self.http_client,
            outcome_address,
            agentic_job_id,
            payload,
            team_id=team_id,
            api_key=self._outcome_key(),
            retry_attempts=self.settings.outcome_retry_attempts,
            retry_initial_seconds=self.settings.outcome_retry_initial_seconds,
            retry_max_seconds=self.settings.outcome_retry_max_seconds,
        )

    def append_outcome_metrics(
        self,
        agentic_job_id: str,
        entries: Sequence[Mapping[str, Any]],
        *,
        dry_run: bool = False,
    ) -> None:
        """Append declared per-job metric facts to an existing job.

        The entries are sent as the request body itself — a bare JSON array,
        which is what this endpoint declares — with each entry forwarded exactly
        as supplied::

            [{"key": "quality_rate", "value": 0.93, "provenance": "MEASURED"}]

        ``provenance``, ``recordedBy`` and ``source`` are optional and default
        server-side (SELF_REPORTED, the calling principal, api). The metric must
        already be declared PER_JOB on the job type's economics contract, and
        quality_rate is range-checked to 0..1 by the server; an undeclared key
        or an out-of-range rate is a 400.

        Unlike ``report_outcome``, whose payload is an opaque dict this client
        deliberately passes through untouched, ``entries`` is a typed argument
        and gets the same shape check ``JobContext`` applies: a sequence of
        mappings, each with a non-blank ``key`` and a ``value``, at least one of
        them. It runs before the dry-run print and before any key or team
        resolution, so both entry points reject the same inputs at the same
        point and a dry run never prints a payload the real call would refuse.

        Raises:
            ValueError: no entries, a malformed entry, or a metering
                (``rev_mk_``) API key.
        """
        validated = _outcomes.normalize_outcome_metrics(
            entries, require_entries=True
        ) or []
        outcome_address = self._outcome_address()
        safe_id = urllib.parse.quote(agentic_job_id, safe="")
        url = (
            f"{outcome_address.rstrip('/')}"
            f"/profitstream/v2/api/jobs/{safe_id}/outcome/metrics"
        )
        if dry_run:
            _print_dry_run("POST", url, validated)
            return
        api_key = self._outcome_key()
        team_id = self._get_team_id()
        _outcomes.append_outcome_metrics_request(
            self.http_client,
            outcome_address,
            agentic_job_id,
            validated,
            team_id=team_id,
            api_key=api_key,
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
        address = self._outcome_address()
        if address in self._resolved_team_ids:
            return self._resolved_team_ids[address]

        # Delegate auto-resolution (key-prefix + ProfitStream teams API) to the
        # shared transport. use_env=False preserves this client's historical
        # chain: settings.team_id > cached auto-resolution — REVENIUM_TEAM_ID
        # is already folded into settings by the examples-pack loader.
        resolved = _outcomes.resolve_team_id(
            "",
            self._outcome_key(),
            self.http_client,
            address,
            use_env=False,
        )
        if resolved:
            self._resolved_team_ids[address] = resolved
        return resolved


def _queue(coroutine: Coroutine[Any, Any, None], *, gated_by_circuit: bool = True) -> None:
    if run_async_in_thread(coroutine, gated_by_circuit=gated_by_circuit) is None:
        # Not queued (shutdown has begun); close it so it is not reported as never awaited.
        coroutine.close()


async def _send_completion(args: Dict[str, Any], idempotency_key: str, transaction_id: Any) -> None:
    try:
        submit_ai_event("completion", args, idempotency_key=idempotency_key)
    except APIError as exc:
        # submit_ai_event has logged it already.
        logger.debug("emit_completion record %s was not delivered: %s", transaction_id, type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - nothing may escape into the metering worker
        logger.error("emit_completion record %s was not sent: %s", transaction_id, type(exc).__name__)


async def _send_tool_event(tool_event: Dict[str, Any], transaction_id: Any) -> None:
    try:
        deliver_tool_event(tool_event)
    except Exception as exc:  # noqa: BLE001 - nothing may escape into the metering worker
        logger.error("emit_tool_event record %s was not sent: %s", transaction_id, type(exc).__name__)


def _split_completion_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Sort a completion payload into typed keywords, extra_body and unknown keys.

    A required key is passed as given, so an explicit ``None`` still reaches
    the wire as null. Everywhere else ``None`` means not set, and the key is
    omitted so the typed client keeps its NotGiven default.
    """
    keywords: Dict[str, Any] = {}
    extra_body: Dict[str, Any] = {}
    unknown_keys: List[str] = []
    for key, value in payload.items():
        if key in _REQUIRED_COMPLETION_KEYS:
            keywords[_COMPLETION_KEYWORD_BY_WIRE_NAME[key]] = value
        elif key not in _EXTRA_BODY_COMPLETION_KEYS and key not in _COMPLETION_KEYWORD_BY_WIRE_NAME:
            unknown_keys.append(str(key))
        elif value is None:
            continue
        elif key in _EXTRA_BODY_COMPLETION_KEYS:
            extra_body[key] = value
        elif key == "agentVersion":
            agent_version = validate_agent_version(value)
            if agent_version is not None:
                keywords["agent_version"] = agent_version
        else:
            keywords[_COMPLETION_KEYWORD_BY_WIRE_NAME[key]] = value
    return keywords, extra_body, unknown_keys


def _job_resource_or(
    request_body: Dict[str, Any], response: Optional[httpx.Response]
) -> Dict[str, Any]:
    """The created job resource merged over ``request_body``.

    Merged, not substituted: this call has always returned every field the
    caller supplied, so a response that omits one (a partial job resource, a
    projection) must not silently drop it — the response wins only where the
    two overlap. The accepted 409 of an idempotent re-run carries an error body
    rather than a job, and a bodiless or non-JSON 2xx carries nothing at all;
    neither is a failure, so both fall back to the request body alone.
    """
    if response is None or response.status_code == 409:
        return request_body
    try:
        data = response.json()
    except Exception:  # noqa: BLE001 — a bodiless or non-JSON 2xx is expected
        return request_body
    if isinstance(data, dict) and data:
        return {**request_body, **data}
    return request_body


def _print_dry_run(
    method: str, url: str, payload: Union[Dict[str, Any], List[Any]]
) -> None:
    print(f"[DRY-RUN] {method} {url}")
    print(json.dumps(payload, indent=2))
