"""The metering context of one fal generation, captured when the caller starts it."""

import datetime
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function, merge_metadata
from ._metering import generate_transaction_id, handle_metering

UNKNOWN_APPLICATION = "unknown"


def is_metering_active() -> bool:
    return not is_selective_metering_enabled() or is_inside_decorated_function()


@dataclass(frozen=True)
class FalCall:
    application: str
    arguments: Mapping[str, Any]
    usage_metadata: Dict[str, Any]
    request_time_dt: datetime.datetime
    transaction_id: str
    metered: bool

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
    def from_request(cls, args: Sequence[Any], kwargs: Dict[str, Any]) -> "FalCall":
        """Start a call from a fal client method's ``(application, arguments, ...)`` parameters.

        ``usage_metadata`` is ours, not fal's, so it is removed from ``kwargs``
        before fal sees them.
        """
        api_metadata = kwargs.pop("usage_metadata", None)
        application = args[0] if args else kwargs.get("application", UNKNOWN_APPLICATION)
        arguments = args[1] if len(args) > 1 else kwargs.get("arguments")
        return cls.start(application, arguments, api_metadata)

    def meter(self, result: Any, is_streamed: bool = False) -> None:
        if not self.metered:
            return
        handle_metering(
            application=self.application,
            arguments=self.arguments,
            result=result,
            request_time_dt=self.request_time_dt,
            usage_metadata=self.usage_metadata,
            transaction_id=self.transaction_id,
            is_streamed=is_streamed,
        )
