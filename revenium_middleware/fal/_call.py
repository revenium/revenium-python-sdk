"""The metering context of one fal generation, captured when the caller starts it."""

import datetime
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence

from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function, merge_metadata
from revenium_middleware._core.selective_metering import reported_if_unmetered
from revenium_middleware._core.enforcement import check_enforcement
from revenium_middleware._core.metering import current_client, recorded_under
from ._metering import generate_transaction_id, handle_metering
from .config import Config
from .trace_fields import normalize_model_name

UNKNOWN_APPLICATION = "unknown"

# The argument keys the extract_*_fields readers in _metering.py bill from. Kept by hand: a key a reader
# starts using must be added here, or queued jobs are billed without it.
BILLED_ARGUMENT_KEYS = ("num_images", "image_size", "quality", "aspect_ratio", "duration", "resolution", "fps",
                        "text", "input")
# Inline audio can be megabytes, and its record reads only whether it was given.
GIVEN_OR_NOT_ARGUMENT_KEYS = ("audio", "audio_url")


def is_metering_active() -> bool:
    return not reported_if_unmetered(is_selective_metering_enabled() and not is_inside_decorated_function())


def request_arguments(args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
    """The ``arguments`` a fal client method was called with, as the caller passed them."""
    return args[1] if len(args) > 1 else kwargs.get("arguments")


def billing_arguments(arguments: Any) -> Dict[str, Any]:
    """The part of a call's ``arguments`` its metering record reads."""
    if not isinstance(arguments, Mapping):
        return {}
    kept = {key: arguments[key] for key in BILLED_ARGUMENT_KEYS if key in arguments}
    kept.update({key: bool(arguments[key]) for key in GIVEN_OR_NOT_ARGUMENT_KEYS if key in arguments})
    return kept


@dataclass(frozen=True)
class FalCall:
    application: str
    arguments: Mapping[str, Any]
    usage_metadata: Dict[str, Any]
    request_time_dt: datetime.datetime
    transaction_id: str
    metered: bool
    metering_client: Any = field(default_factory=current_client)

    @classmethod
    def start(cls, application: str, arguments: Optional[Mapping[str, Any]] = None,
              api_metadata: Optional[Dict[str, Any]] = None) -> "FalCall":
        return cls(
            application=application,
            arguments=arguments or {},
            usage_metadata=merge_metadata(api_metadata or {}),
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            transaction_id=generate_transaction_id(),
            metered=is_metering_active(),
        )

    @classmethod
    def from_request(cls, args: Sequence[Any], kwargs: Dict[str, Any], already_checked: bool = False) -> "FalCall":
        """Start a call from a fal client method's ``(application, arguments, ...)`` parameters,
        once a breached cap has had the chance to refuse it, unless the caller already checked it.

        ``usage_metadata`` is ours, not fal's, so it is removed from ``kwargs``
        before fal sees them.
        """
        api_metadata = kwargs.pop("usage_metadata", None)
        application = args[0] if args else kwargs.get("application", UNKNOWN_APPLICATION)
        call = cls.start(application, request_arguments(args, kwargs), api_metadata)
        if not already_checked:
            check_enforcement(call.usage_metadata, model=normalize_model_name(application), provider=Config.PROVIDER)
        return call

    def awaiting_result(self) -> "FalCall":
        """This call as a queued job keeps it until its result is fetched, without the inputs it does not bill."""
        return replace(self, arguments=billing_arguments(self.arguments))

    def meter(self, result: Any, is_streamed: bool = False) -> None:
        if not self.metered:
            return
        with recorded_under(self.metering_client):
            handle_metering(
                application=self.application,
                arguments=self.arguments,
                result=result,
                request_time_dt=self.request_time_dt,
                usage_metadata=self.usage_metadata,
                transaction_id=self.transaction_id,
                is_streamed=is_streamed,
            )
