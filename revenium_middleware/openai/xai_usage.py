"""How xAI bills a call, read from its OpenAI-compatible usage.

xAI's reasoning models report ``completion_tokens`` as the answer alone and
bill the reasoning tokens as output beside it, so ``total_tokens`` is prompt +
completion + reasoning. xAI also reports its own charge for the call as
``cost_in_usd_ticks``.
"""
import math
from typing import Any, Dict

from revenium_middleware._core.cache_tokens import get_usage_field

USD_TICKS_PER_DOLLAR = 10_000_000_000


def billed_output_tokens(prompt_tokens: int, completion_tokens: int, total_tokens: Any, reasoning_tokens: int) -> int:
    """The output tokens xAI bills: the answer plus the reasoning, when its total says the
    reasoning sits outside ``completion_tokens``."""
    if reasoning_tokens > 0 and total_tokens == prompt_tokens + completion_tokens + reasoning_tokens:
        return completion_tokens + reasoning_tokens
    return completion_tokens


def reported_cost(usage: Any) -> Dict[str, float]:
    """xAI's own charge for the call as the record's total cost; empty when it reported none."""
    ticks = get_usage_field(usage, "cost_in_usd_ticks", None)
    if isinstance(ticks, bool) or not isinstance(ticks, (int, float)) or not math.isfinite(ticks) or ticks < 0:
        return {}
    return {"total_cost": ticks / USD_TICKS_PER_DOLLAR}
