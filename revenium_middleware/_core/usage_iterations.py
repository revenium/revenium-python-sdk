"""Usage records for the passes Anthropic bills outside a response's top-level usage.

An advisor sub-inference and a compaction pass each appear as one entry of
``usage.iterations`` and are not counted in ``usage.input_tokens`` /
``usage.output_tokens``. Each becomes its own completion record, priced at the
model that ran it and linked to the call's main record.

Lives in ``_core`` so the OpenAI/LangChain callback can bill passes without
loading the Anthropic integration: never import ``revenium_middleware.anthropic``
from here, since that import installs the Anthropic transport.
"""
import hashlib
import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from revenium_middleware._core.cache_tokens import _get, extract_cache_creation_ttl_counts, get_usage_field
from revenium_middleware._core.context import _MAX_IDEMPOTENCY_KEY_LENGTH

logger = logging.getLogger("revenium_middleware")

ADVISOR_MESSAGE = "advisor_message"
COMPACTION = "compaction"
SEPARATELY_BILLED_ITERATIONS = (ADVISOR_MESSAGE, COMPACTION)

_FIVE_MINUTE_KEY = "cache_creation5m_token_count"
_ONE_HOUR_KEY = "cache_creation1h_token_count"
_CACHE_TTL_KEYS = (_FIVE_MINUTE_KEY, _ONE_HOUR_KEY)
_PROMPT_CAPTURE_CLEARED = {
    "system_prompt": None,
    "input_messages": None,
    "output_response": None,
    "prompts_truncated": None,
}


def iteration_id(main_id: str, index: int) -> str:
    return f"{main_id}-iteration-{index}"


def iteration_idempotency_key(main_key: str, index: int) -> str:
    derived = iteration_id(main_key, index)
    if len(derived) <= _MAX_IDEMPOTENCY_KEY_LENGTH:
        return derived
    return iteration_id(hashlib.sha256(main_key.encode()).hexdigest(), index)


def _iterations(usage_or_iterations: Any) -> List[Any]:
    if isinstance(usage_or_iterations, list):
        return usage_or_iterations
    iterations = _get(usage_or_iterations, "iterations")
    return iterations if isinstance(iterations, list) else []


def _is_billed(entry: Any) -> bool:
    return _get(entry, "type") in SEPARATELY_BILLED_ITERATIONS


def billed_iterations(usage_or_iterations: Any) -> List[Tuple[int, Any]]:
    """The separately billed entries with their position in the full iterations list."""
    return [(index, entry) for index, entry in enumerate(_iterations(usage_or_iterations)) if _is_billed(entry)]


class IterationUsage(NamedTuple):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def gross_prompt_tokens(self) -> int:
        """The prompt count a LiteLLM-style usage reports: input plus both cache buckets."""
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


def _summed_usage(entries: List[Any]) -> IterationUsage:
    return IterationUsage(*(
        sum(get_usage_field(entry, name) for entry in entries)
        for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    ))


def billed_iteration_usage(usage_or_iterations: Any) -> IterationUsage:
    """The tokens of every separately billed pass, summed."""
    return _summed_usage([entry for _, entry in billed_iterations(usage_or_iterations)])


TOTALS_FOLD_EVERY_ITERATION = "folded"
TOTALS_ARE_EXECUTOR_ONLY = "executor_only"
TOTALS_UNRECOGNISED = "unknown"


def iteration_totals_shape(prompt_tokens: int, completion_tokens: int, usage_or_iterations: Any) -> str:
    """Which iterations a reported prompt and completion count already contain.

    LiteLLM sums every iteration into its totals; Anthropic's own top level
    holds the executor's passes only. Any other shape is unrecognised, and a
    caller must then neither net the totals nor bill the passes again.
    """
    iterations = _iterations(usage_or_iterations)
    reported = (prompt_tokens, completion_tokens)
    every_pass = _summed_usage(iterations)
    if reported == (every_pass.gross_prompt_tokens, every_pass.output_tokens):
        return TOTALS_FOLD_EVERY_ITERATION
    executor = _summed_usage([entry for entry in iterations if not _is_billed(entry)])
    if reported == (executor.gross_prompt_tokens, executor.output_tokens):
        return TOTALS_ARE_EXECUTOR_ONLY
    return TOTALS_UNRECOGNISED


def executor_cache_creation_ttl_counts(usage_or_iterations: Any) -> Dict[str, int]:
    """The 5m/1h cache-write split of the executor's passes, or {} when no pass reports one.

    Applies LiteLLM's own aggregation rule to the passes that are not billed
    separately: the detailed buckets summed, and the part of their write no
    breakdown accounts for counted as five-minute.
    """
    iterations = _iterations(usage_or_iterations)
    if not any(extract_cache_creation_ttl_counts(entry) for entry in iterations):
        return {}
    executor = [entry for entry in iterations if not _is_billed(entry)]
    five_minute = sum(extract_cache_creation_ttl_counts(entry).get(_FIVE_MINUTE_KEY, 0) for entry in executor)
    one_hour = sum(extract_cache_creation_ttl_counts(entry).get(_ONE_HOUR_KEY, 0) for entry in executor)
    undetailed = max(_summed_usage(executor).cache_creation_tokens - five_minute - one_hour, 0)
    return {_FIVE_MINUTE_KEY: five_minute + undetailed, _ONE_HOUR_KEY: one_hour}


def _advisor_tool_model(request_kwargs: Optional[Dict[str, Any]]) -> Optional[str]:
    for tool in (request_kwargs or {}).get("tools") or []:
        if str(_get(tool, "type") or "").startswith("advisor_"):
            return _get(tool, "model")
    return None


def _iteration_model(entry: Any, main_args: Dict[str, Any], request_kwargs: Optional[Dict[str, Any]]) -> Any:
    if _get(entry, "type") == ADVISOR_MESSAGE:
        return _get(entry, "model") or _advisor_tool_model(request_kwargs) or main_args["model"]
    return (request_kwargs or {}).get("model") or main_args["model"]


def iteration_record(main_args: Dict[str, Any], index: int, entry: Any,
                     request_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The main record's payload, re-scoped to one billed iteration."""
    input_tokens = get_usage_field(entry, "input_tokens")
    output_tokens = get_usage_field(entry, "output_tokens")
    record = {key: value for key, value in main_args.items() if key not in _CACHE_TTL_KEYS}
    record.update(_PROMPT_CAPTURE_CLEARED)
    record.update({
        "transaction_id": iteration_id(main_args["transaction_id"], index),
        "parent_transaction_id": main_args["transaction_id"],
        "operation_subtype": _get(entry, "type"),
        "model": _iteration_model(entry, main_args, request_kwargs),
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "total_token_count": input_tokens + output_tokens,
        "cache_creation_token_count": get_usage_field(entry, "cache_creation_input_tokens"),
        "cache_read_token_count": get_usage_field(entry, "cache_read_input_tokens"),
        **extract_cache_creation_ttl_counts(entry),
        "reasoning_token_count": 0,
        "stop_reason": "END",
    })
    return record


def submit_iteration_records(submit: Callable[..., Any], main_args: Optional[Dict[str, Any]],
                             usage_or_iterations: Any, request_kwargs: Optional[Dict[str, Any]],
                             main_idempotency_key: Optional[str]) -> None:
    """One record per advisor or compaction pass, sent even when the main record failed.

    Each takes an explicit Idempotency-Key derived from the main record's: left to
    submit_ai_event, the extras would draw from an idempotency_key() block's counter
    and shift the key of every later call in the block.

    ``submit`` is the caller's own submit_ai_event, passed in so each integration
    keeps sending through the name its module holds.
    """
    if main_args is None:
        return
    for index, entry in billed_iterations(usage_or_iterations):
        transaction_id = iteration_id(main_args["transaction_id"], index)
        key = iteration_idempotency_key(main_idempotency_key, index) if main_idempotency_key else transaction_id
        try:
            submit("completion", iteration_record(main_args, index, entry, request_kwargs), idempotency_key=key)
        except Exception as e:
            logger.warning("Metering call for %s failed: %s", transaction_id, e)
