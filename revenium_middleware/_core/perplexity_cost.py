"""The charge Perplexity reports in ``usage.cost``, mapped to the metering cost fields.

Perplexity's ``total_cost`` already includes its flat per-request fee, which no token count
can rebuild, so it is forwarded as reported and never summed from the parts.

This module lives in ``_core`` because importing ``revenium_middleware.perplexity`` or
``revenium_middleware.openai`` patches provider clients, and both middlewares need it.
"""
import inspect
import math
from typing import Any, Callable, Dict

# Metering field -> Perplexity's names for it: chat completions first, then the Responses API.
_REPORTED_COST_FIELDS = {
    "input_token_cost": ("input_tokens_cost", "input_cost"),
    "output_token_cost": ("output_tokens_cost", "output_cost"),
    "total_cost": ("total_cost",),
}


def _cost_value(cost: Any, name: str) -> Any:
    if isinstance(cost, dict):
        return cost.get(name)
    return getattr(cost, name, None)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _usage_cost(carrier: Any) -> Any:
    usage = getattr(carrier, 'usage', None)
    return getattr(usage, 'cost', None) if usage else None


def reported_cost_of(item: Any) -> Any:
    """The ``usage.cost`` Perplexity attached to a response, a stream chunk or
    a Responses stream event (whose response carries it), or None."""
    cost = _usage_cost(item)
    return cost if cost is not None else _usage_cost(getattr(item, 'response', None))


def reported_cost_fields(usage: Any) -> Dict[str, float]:
    """Perplexity's own charge in ``usage.cost``, keyed by metering field; empty when it reported none.
    LiteLLM flattens the cost on a stream's usage chunk to the bare total."""
    cost = getattr(usage, 'cost', None) if usage else None
    if _is_finite_number(cost):
        return {"total_cost": float(cost)}
    if cost is None:
        return {}
    reported = {}
    for field, sources in _REPORTED_COST_FIELDS.items():
        value = next((value for value in (_cost_value(cost, source) for source in sources)
                      if _is_finite_number(value)), None)
        if value is not None:
            reported[field] = float(value)
    return reported


def extract_reported_cost(response: Any) -> Dict[str, float]:
    """Perplexity's own charge for the call, keyed by metering field; empty when it reported none."""
    return reported_cost_fields(getattr(response, 'usage', None))


class ReportedCostReadingStream:
    """Passes a Perplexity stream through unchanged, handing each item to ``observe`` as it goes by."""

    def __init__(self, stream: Any, observe: Callable[[Any], None]):
        self._stream = stream
        self._observe = observe
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self._stream)
        item = next(self._iterator)
        self._observe(item)
        return item

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._stream.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class AsyncReportedCostReadingStream:
    """The async form of ``ReportedCostReadingStream``."""

    def __init__(self, stream: Any, observe: Callable[[Any], None]):
        self._stream = stream
        self._observe = observe
        self._iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()
        item = await self._iterator.__anext__()
        self._observe(item)
        return item

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info):
        return await self._stream.__aexit__(*exc_info)

    async def aclose(self):
        """Closes the wrapped stream. openai's ``AsyncStream`` has only ``close``, and a
        caller that stops early (langchain-openai's chunk timeout) looks for ``aclose``."""
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if close is None:
            return
        closed = close()
        if inspect.isawaitable(closed):
            await closed

    def __getattr__(self, name):
        return getattr(self._stream, name)


def read_reported_usage(response: Any, observe: Callable[[Any], None]) -> Any:
    """``response`` with each chunk or event it yields handed to ``observe``: a stream comes
    back wrapped, and anything else is observed at once and returned as it is."""
    if hasattr(response, "__next__"):
        return ReportedCostReadingStream(response, observe)
    if hasattr(response, "__anext__"):
        return AsyncReportedCostReadingStream(response, observe)
    observe(response)
    return response
