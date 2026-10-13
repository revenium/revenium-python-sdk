"""Which model a Bedrock application inference profile or Provisioned Throughput runs.

Neither ARN names its model, so the Bedrock transport asks the Bedrock
control plane (``GetInferenceProfile`` or ``GetProvisionedModelThroughput``)
once per ARN per process and shares the answer with every caller. The transport
looks a profile up in the background and never holds a caller's call for it:
calls made before the lookup answers are neither metered nor capped. A lookup
that fails never fails the caller's call: the profile is left unmetered and
one warning says so. A failure is remembered only for the credentials that
met it, so a caller allowed to look the profile up still makes its own lookup.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Hashable, Optional, Set, Tuple

logger = logging.getLogger("revenium_middleware")

APPLICATION_PROFILE_ARN = re.compile(
    r"^arn:aws[a-z-]*:bedrock:([a-z0-9-]+):(\d{12}):application-inference-profile/[^/\s]+$"
)
PROVISIONED_MODEL_ARN = re.compile(
    r"^arn:aws[a-z-]*:bedrock:([a-z0-9-]+):(\d{12}):provisioned-model/[^/\s]+$"
)
_CONTEXT_WINDOW_SUFFIX = re.compile(r":\d+k$")

_TRANSIENT_RETRY_SECONDS = 300
_DEFINITE_ERROR_CODES = frozenset({"AccessDeniedException", "ResourceNotFoundException", "ValidationException"})
_FOUNDATION_MODEL = "foundation-model/"
_ACCESS_DENIED = "AccessDeniedException"
_ARN_REGION_FIELD = 3
_US_REGION, _US_GOV_REGION, _CANADA_REGION, _EU_REGION = "us-", "us-gov-", "ca-", "eu-"
_US_GEOGRAPHY, _EU_GEOGRAPHY = "us", "eu"
_LOOKUP_THREAD_NAME = "revenium-bedrock-model-lookup"


def is_application_profile_arn(model_id: Any) -> bool:
    return isinstance(model_id, str) and APPLICATION_PROFILE_ARN.match(model_id) is not None


def is_routed_arn(model_id: Any) -> bool:
    """An application inference profile or Provisioned Throughput ARN: one that does not name its model."""
    return is_application_profile_arn(model_id) or (
        isinstance(model_id, str) and PROVISIONED_MODEL_ARN.match(model_id) is not None)


@dataclass(frozen=True)
class UnresolvedProfileEffect:
    """What a failed lookup costs the caller's calls, as the warning words it."""
    consequence: str
    remedy: str


UNMETERED = UnresolvedProfileEffect("its calls are neither metered nor capped", "meter them")
NOT_CAPPED_BY_MODEL = UnresolvedProfileEffect(
    "its AnthropicBedrock calls are metered but cannot be capped by model", "cap them by model")


class AnthropicBedrockCredentials:
    """What the control-plane lookup reads from a botocore runtime client,
    taken from an ``AnthropicBedrock`` client's own settings."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.meta = SimpleNamespace(region_name=getattr(client, "aws_region", None))

    def credential_identity(self) -> Hashable:
        """Where the client's credentials come from, read from its settings alone."""
        client = self._client
        return ("api_key" if getattr(client, "api_key", None) is not None else "aws",
                getattr(client, "aws_profile", None), getattr(client, "aws_access_key", None),
                getattr(client, "aws_region", None))

    def _get_credentials(self) -> Any:
        if getattr(self._client, "api_key", None) is not None:
            return None
        import boto3

        return boto3.Session(
            profile_name=getattr(self._client, "aws_profile", None),
            region_name=getattr(self._client, "aws_region", None),
            aws_access_key_id=getattr(self._client, "aws_access_key", None),
            aws_secret_access_key=getattr(self._client, "aws_secret_key", None),
            aws_session_token=getattr(self._client, "aws_session_token", None),
        ).get_credentials()


class _Unresolvable(Exception):
    """The profile cannot be resolved; the message is ours and safe to log."""


@dataclass(frozen=True)
class _Entry:
    claude_model_id: Optional[str] = None
    failed: bool = False
    retry_at: Optional[float] = None

    def is_current(self, now: float) -> bool:
        return self.retry_at is None or now < self.retry_at


_CALLER_NETWORK_SETTINGS = ("proxies", "proxies_config", "client_cert")


def _control_plane_client(runtime_client: Any, region: str) -> Any:
    """A ``bedrock`` client in the profile's region, signed with the caller's
    credentials and reaching AWS through the caller's proxy, client
    certificate and CA bundle. Not the runtime client's endpoint: that serves
    bedrock-runtime. Its own short timeouts and single attempt stay, so a
    lookup never inherits the caller's retries."""
    import botocore.session
    from botocore.config import Config

    get_credentials = getattr(runtime_client, "_get_credentials", None)
    credentials = get_credentials() if get_credentials is not None else None
    if credentials is None:
        raise _Unresolvable("no credentials")
    frozen = credentials.get_frozen_credentials()
    caller_config = getattr(getattr(runtime_client, "meta", None), "config", None)
    network = {name: getattr(caller_config, name, None) for name in _CALLER_NETWORK_SETTINGS}
    verify = _resolved_verify(runtime_client)
    return botocore.session.get_session().create_client(
        "bedrock",
        region_name=region,
        aws_access_key_id=frozen.access_key,
        aws_secret_access_key=frozen.secret_key,
        aws_session_token=frozen.token,
        config=Config(connect_timeout=2, read_timeout=3, retries={"total_max_attempts": 1, "mode": "standard"},
                      **network),
        **({} if verify is None else {"verify": verify}),
    )


def _resolved_verify(runtime_client: Any) -> Any:
    """The CA bundle (or ``False``) the caller's client verifies TLS with, or
    None when it cannot be read. botocore keeps the resolved value only in
    private state, so a rename there falls back to botocore's own default."""
    http_session = getattr(getattr(runtime_client, "_endpoint", None), "http_session", None)
    return getattr(http_session, "_verify", None)


def _routed_model_id(response: Dict[str, Any]) -> str:
    """The one foundation model id a profile routes to, as the US or EU
    cross-region profile it was copied from when it was. A profile copied
    from a cross-region profile lists the same model once per region."""
    model_ids, regions = set(), set()
    for model in response.get("models") or []:
        model_arn = model.get("modelArn") or ""
        if _FOUNDATION_MODEL not in model_arn:
            raise _Unresolvable("unreadable profile")
        model_ids.add(model_arn.split(_FOUNDATION_MODEL, 1)[1])
        regions.add(_arn_region(model_arn))
    if len(model_ids) != 1:
        raise _Unresolvable("routes to more than one model" if model_ids else "names no model")
    model_id = model_ids.pop()
    geography = _priced_geography(regions)
    return f"{geography}.{model_id}" if geography else model_id


def _application_profile_model_id(client: Any, profile_arn: str) -> str:
    return _routed_model_id(client.get_inference_profile(inferenceProfileIdentifier=profile_arn))


def _provisioned_model_id(client: Any, provisioned_arn: str) -> str:
    """The foundation model a Provisioned Throughput was bought for, without
    the context-window tail (``:200k``) its model id may carry. A Provisioned
    Throughput serves one Region, so it has no geography."""
    response = client.get_provisioned_model_throughput(provisionedModelId=provisioned_arn)
    foundation_model_arn = response.get("foundationModelArn") or ""
    if _FOUNDATION_MODEL not in foundation_model_arn:
        raise _Unresolvable("unreadable provisioned model")
    return _CONTEXT_WINDOW_SUFFIX.sub("", foundation_model_arn.split(_FOUNDATION_MODEL, 1)[1])


@dataclass(frozen=True)
class _RoutedArnKind:
    """An ARN that does not name its model: how the messages name it, the
    permission its lookup needs, and the lookup itself."""
    pattern: Any
    subject: str
    verb: str
    caveat: str
    permission: str
    look_up: Callable[[Any, str], str]


_APPLICATION_PROFILE = _RoutedArnKind(
    APPLICATION_PROFILE_ARN, "application inference profile", "routes to", "",
    "bedrock:GetInferenceProfile", _application_profile_model_id)
_PROVISIONED_THROUGHPUT = _RoutedArnKind(
    PROVISIONED_MODEL_ARN, "Provisioned Throughput", "runs", "if it runs Claude, ",
    "bedrock:GetProvisionedModelThroughput", _provisioned_model_id)


def _kind(routed_arn: str) -> _RoutedArnKind:
    return _PROVISIONED_THROUGHPUT if PROVISIONED_MODEL_ARN.match(routed_arn) else _APPLICATION_PROFILE


def _arn_region(arn: str) -> str:
    """Empty for the region-less ARN a global profile lists."""
    fields = arn.split(":")
    return fields[_ARN_REGION_FIELD] if len(fields) > _ARN_REGION_FIELD else ""


def _is_us_region(region: str) -> bool:
    return region.startswith(_US_REGION) and not region.startswith(_US_GOV_REGION)


def _priced_geography(regions: Set[str]) -> Optional[str]:
    """The geography a profile spanning ``regions`` is priced as, if any.

    Only US and EU: the platform prices an unknown geography id at zero, and
    its price table lacks Australia, Japan and APAC ids for several Claude
    models. The US profile called from Canada also lists a Canadian region.
    """
    if len(regions) < 2:
        return None
    if any(map(_is_us_region, regions)) and all(
            _is_us_region(region) or region.startswith(_CANADA_REGION) for region in regions):
        return _US_GEOGRAPHY
    if all(region.startswith(_EU_REGION) for region in regions):
        return _EU_GEOGRAPHY
    return None


def _failure(exc: Exception) -> Tuple[str, bool]:
    """A loggable reason (never the raw message, which may carry a signed
    URL) and whether the failure is worth one more try later. Only a definite
    answer is final: an outage, a dropped connection or an expired session
    token clears by itself."""
    if isinstance(exc, _Unresolvable):
        return str(exc), False
    error = getattr(exc, "response", None)
    code = error.get("Error", {}).get("Code") if isinstance(error, dict) else None
    if code:
        return code, code not in _DEFINITE_ERROR_CODES
    return type(exc).__name__, True


def _credential_identity(runtime_client: Any) -> Hashable:
    """Which credentials a lookup through ``runtime_client`` would sign with, without resolving any."""
    credential_identity = getattr(runtime_client, "credential_identity", None)
    if credential_identity is not None:
        return credential_identity()
    get_credentials = getattr(runtime_client, "_get_credentials", None)
    credentials = get_credentials() if get_credentials is not None else None
    return getattr(credentials, "access_key", None)


def _region(profile_arn: str, runtime_client: Any) -> Optional[str]:
    match = _kind(profile_arn).pattern.match(profile_arn)
    if match:
        return match.group(1)
    return getattr(getattr(runtime_client, "meta", None), "region_name", None)


def _start_lookup_thread(job: Callable[[], None]) -> None:
    """A daemon thread, so a lookup that never answers never holds the host application at exit."""
    threading.Thread(target=job, name=_LOOKUP_THREAD_NAME, daemon=True).start()


_PENDING = _Entry()


class InferenceProfileResolver:
    """Resolves each application inference profile or Provisioned Throughput
    ARN (any ARN that does not name its model) once, thread-safely, and
    shares the answer. A caller that waits resolves the profile itself; one
    that does not starts at most one background lookup per profile and
    credential identity, and is answered from the cache until it lands.
    Lookups on other profiles are not held up. A failed lookup is cached per
    credential identity, so it never stops other credentials trying."""

    def __init__(self, *, is_claude: Callable[[str], bool],
                 client_factory: Callable[[Any, Optional[str]], Any] = _control_plane_client,
                 start_lookup: Callable[[Callable[[], None]], None] = _start_lookup_thread) -> None:
        self._is_claude = is_claude
        self._client_factory = client_factory
        self._start_lookup = start_lookup
        self._lock = threading.Lock()
        self._resolved: Dict[str, _Entry] = {}
        self._failed: Dict[Tuple[str, Hashable], _Entry] = {}
        self._arn_locks: Dict[str, threading.Lock] = {}
        self._in_flight: Set[Tuple[str, Hashable]] = set()
        self._warned: Set[Tuple[str, UnresolvedProfileEffect]] = set()
        self._pending_noted: Set[str] = set()

    def claude_model_id(self, runtime_client: Any, profile_arn: str, *,
                        unresolved: UnresolvedProfileEffect = UNMETERED, wait: bool = True) -> Optional[str]:
        """The Bedrock model id the profile routes to when it is Claude, else
        None. Without ``wait``, a profile not yet looked up answers None while
        its lookup runs in the background."""
        try:
            return self._entry(runtime_client, profile_arn, unresolved, wait=wait).claude_model_id
        except Exception as exc:  # noqa: BLE001 - never break the caller's request
            logger.debug("Bedrock inference profile resolution error: %s", type(exc).__name__)
            return None

    def forget_in_flight(self) -> None:
        """Give a forked child fresh locks and no lookups in flight: the
        parent's lookup threads do not exist in the child, so a profile they
        were resolving would otherwise never be looked up there."""
        self._lock = threading.Lock()
        self._arn_locks = {}
        self._in_flight = set()

    def cached_claude_model_id(self, profile_arn: str) -> Optional[str]:
        """What ``claude_model_id`` would answer from the cache alone, never looking the profile up."""
        entry = self._shared(profile_arn)
        return entry.claude_model_id if entry is not None else None

    def _entry(self, runtime_client: Any, profile_arn: str, unresolved: UnresolvedProfileEffect, *,
               wait: bool) -> _Entry:
        entry = self._shared(profile_arn)
        if entry is not None:
            return entry
        failure_key = (profile_arn, _credential_identity(runtime_client))
        entry = self._current(failure_key)
        if entry is not None:
            return entry
        if wait:
            return self._resolve_once(runtime_client, profile_arn, unresolved, failure_key)
        return self._pending_after_starting_lookup(runtime_client, profile_arn, unresolved, failure_key)

    def _pending_after_starting_lookup(self, runtime_client: Any, profile_arn: str,
                                       unresolved: UnresolvedProfileEffect,
                                       failure_key: Tuple[str, Hashable]) -> _Entry:
        self._look_up_in_background(runtime_client, profile_arn, unresolved, failure_key)
        entry = self._current(failure_key)
        if entry is not None:
            return entry
        self._note_pending(profile_arn)
        return _PENDING

    def _resolve_once(self, runtime_client: Any, profile_arn: str, unresolved: UnresolvedProfileEffect,
                      failure_key: Tuple[str, Hashable]) -> _Entry:
        with self._arn_lock(profile_arn):
            entry = self._current(failure_key)
            if entry is None:
                entry = self._resolve(runtime_client, profile_arn, unresolved)
                self._store(failure_key, entry)
        return entry

    def _look_up_in_background(self, runtime_client: Any, profile_arn: str, unresolved: UnresolvedProfileEffect,
                               failure_key: Tuple[str, Hashable]) -> None:
        with self._lock:
            if failure_key in self._in_flight or self._current_locked(failure_key) is not None:
                return
            self._in_flight.add(failure_key)

        def job() -> None:
            try:
                self._resolve_once(runtime_client, profile_arn, unresolved, failure_key)
            except Exception as exc:  # noqa: BLE001 - a background lookup must never raise
                logger.debug("Bedrock inference profile lookup error: %s", type(exc).__name__)
            finally:
                self._finish_lookup(failure_key)

        try:
            self._start_lookup(job)
        except Exception as exc:  # noqa: BLE001 - e.g. no new threads at interpreter shutdown
            self._finish_lookup(failure_key)
            logger.debug("Bedrock inference profile lookup could not start: %s", type(exc).__name__)

    def _finish_lookup(self, failure_key: Tuple[str, Hashable]) -> None:
        with self._lock:
            self._in_flight.discard(failure_key)

    def _shared(self, profile_arn: str) -> Optional[_Entry]:
        with self._lock:
            return self._resolved.get(profile_arn)

    def _current(self, failure_key: Tuple[str, Hashable]) -> Optional[_Entry]:
        with self._lock:
            return self._current_locked(failure_key)

    def _current_locked(self, failure_key: Tuple[str, Hashable]) -> Optional[_Entry]:
        entry = self._resolved.get(failure_key[0])
        if entry is None:
            entry = self._failed.get(failure_key)
        if entry is not None and entry.is_current(time.monotonic()):
            return entry
        return None

    def _store(self, failure_key: Tuple[str, Hashable], entry: _Entry) -> None:
        with self._lock:
            if entry.failed:
                self._failed[failure_key] = entry
            else:
                self._resolved[failure_key[0]] = entry

    def _arn_lock(self, profile_arn: str) -> threading.Lock:
        with self._lock:
            return self._arn_locks.setdefault(profile_arn, threading.Lock())

    def _resolve(self, runtime_client: Any, profile_arn: str, unresolved: UnresolvedProfileEffect) -> _Entry:
        kind = _kind(profile_arn)
        try:
            client = self._client_factory(runtime_client, _region(profile_arn, runtime_client))
            model_id = kind.look_up(client, profile_arn)
        except Exception as exc:  # noqa: BLE001 - a failed lookup leaves the profile unmetered
            reason, transient = _failure(exc)
            self._warn_once(kind, profile_arn, reason, transient, unresolved)
            return _Entry(failed=True, retry_at=time.monotonic() + _TRANSIENT_RETRY_SECONDS if transient else None)
        if self._is_claude(model_id):
            self._note_recovery(kind, profile_arn, model_id)
            return _Entry(claude_model_id=model_id)
        logger.debug("Bedrock %s %s %s %s, which is not metered here", kind.subject, profile_arn, kind.verb, model_id)
        return _Entry()

    def _warn_once(self, kind: _RoutedArnKind, profile_arn: str, reason: str, transient: bool,
                   unresolved: UnresolvedProfileEffect) -> None:
        with self._lock:
            if (profile_arn, unresolved) in self._warned:
                return
            self._warned.add((profile_arn, unresolved))
        logger.warning(
            "Revenium cannot tell which model the Bedrock %s %s %s (%s); %s%s%s",
            kind.subject, profile_arn, kind.verb, reason, kind.caveat, unresolved.consequence,
            _remedy(reason, transient, unresolved.remedy, kind.permission),
        )

    def _note_pending(self, profile_arn: str) -> None:
        with self._lock:
            if profile_arn in self._pending_noted:
                return
            self._pending_noted.add(profile_arn)
        kind = _kind(profile_arn)
        logger.info("Revenium is looking up which model the Bedrock %s %s %s; "
                    "%sits calls are neither metered nor capped until the lookup answers",
                    kind.subject, profile_arn, kind.verb, kind.caveat)

    def _note_recovery(self, kind: _RoutedArnKind, profile_arn: str, model_id: str) -> None:
        with self._lock:
            warned = {key for key in self._warned if key[0] == profile_arn}
            if not warned:
                return
            self._warned -= warned
        logger.info("Bedrock %s %s now resolves to %s; its calls are metered and capped",
                    kind.subject, profile_arn, model_id)


def _remedy(reason: str, transient: bool, action: str, permission: str) -> str:
    if transient:
        return f" until a lookup succeeds. Revenium retries the lookup every {_TRANSIENT_RETRY_SECONDS // 60} minutes."
    if reason == _ACCESS_DENIED:
        return f". Grant {permission} to {action}."
    return "."
