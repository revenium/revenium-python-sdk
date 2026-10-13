"""Tolerant field access for provider usage objects (dict or attribute-style).

Cache token fields differ by provider surface:

- Anthropic-style: top-level ``cache_read_input_tokens`` / ``cache_creation_input_tokens``,
  plus the nested per-TTL breakdown under ``cache_creation``.
- OpenAI-style: nested ``prompt_tokens_details.cached_tokens`` for reads and
  ``cache_write_tokens`` (``input_tokens_details`` on the Responses API) for
  writes, both counted inside the prompt.

See BACK-2391 for the history of this bug class (hardcoded/zeroed cache-token
fields recurring across multiple emitters) and BACK-1925 for the LiteLLM
client's cache-token contract, which this module's defaults must not break.
"""
import logging
from numbers import Number
from typing import Any, Dict, NamedTuple, Optional

from revenium_middleware._core.package_versions import installed_version, release_of

logger = logging.getLogger("revenium_middleware")


class CacheTokens(NamedTuple):
    cache_read_tokens: int
    cache_creation_tokens: int


def _get(source: Any, name: str) -> Any:
    """Read `name` from a dict or an attribute-style object; None if absent."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def get_usage_field(usage: Any, name: str, default: int = 0) -> Any:
    """Read a single usage field through the same tolerant accessor as
    `extract_cache_tokens`, so cache fields and base token fields (prompt/
    completion/total) are read consistently from the same dict-or-object
    usage value. Missing or None fields return `default`.

    Normalizing a usage object to a plain dict *before* reading it (e.g. to
    simplify a `.get(...)` call) silently drops every field for
    attribute-style usage objects that fall through that normalization --
    exactly the inconsistency this function exists to prevent. Callers with
    multiple fields to read from the same `usage` value should read all of
    them through this function (or `extract_cache_tokens`) rather than
    normalizing `usage` itself first.
    """
    value = _get(usage, name)
    return default if value is None else value


def extract_cache_tokens(usage: Any) -> CacheTokens:
    """Extract (cache_read_tokens, cache_creation_tokens) from a usage object.

    Tolerates dicts and attribute-style objects (SimpleNamespace, SDK response
    models) and treats missing or None fields as 0. Checked in order:

    1. OpenAI-style nested cache reads: ``usage.prompt_tokens_details.cached_tokens``
    2. Anthropic-style top-level cache reads: ``usage.cache_read_input_tokens``
    3. Anthropic-style top-level cache creation: ``usage.cache_creation_input_tokens``
    4. OpenAI-style nested cache writes: see ``openai_cache_write_tokens``
    """
    prompt_details = _get(usage, "prompt_tokens_details")
    cache_read_tokens = get_usage_field(prompt_details, "cached_tokens", 0)
    if not cache_read_tokens:
        cache_read_tokens = get_usage_field(usage, "cache_read_input_tokens", 0)
    cache_creation_tokens = get_usage_field(usage, "cache_creation_input_tokens", 0)
    if not cache_creation_tokens:
        cache_creation_tokens = openai_cache_write_tokens(usage)

    return CacheTokens(cache_read_tokens=cache_read_tokens, cache_creation_tokens=cache_creation_tokens)


def openai_cache_write_tokens(usage: Any) -> int:
    """The prompt tokens OpenAI reports as written to its cache, or 0.

    OpenAI bills a write at its own rate (1.25x input on GPT-5.6 and later)
    and counts it inside ``prompt_tokens`` / ``input_tokens``. Absent, None,
    non-numeric and non-positive values all mean no write.

    Usage that carries a top-level ``cache_creation_input_tokens`` is
    Anthropic-shaped: LiteLLM mirrors an Anthropic or Bedrock write into the
    nested ``cache_write_tokens`` as well, and that write is already owned by
    the top-level count and ``billable_input_tokens``. Reading the mirror would
    take it out of the prompt a second time.
    """
    if _get(usage, "cache_creation_input_tokens") is not None:
        return 0
    for details_name in ("input_tokens_details", "prompt_tokens_details"):
        written = _optional_token_count(_get(usage, details_name), "cache_write_tokens")
        if written is not None and written > 0:
            return written
    return 0


def without_openai_cache_writes(prompt_tokens: int, usage: Any) -> int:
    """``prompt_tokens`` less the OpenAI cache write it contains, never below 0.

    Cache reads stay in: the platform removes them itself for OpenAI-shaped
    rows (see ``billable_input_tokens``).
    """
    written = openai_cache_write_tokens(usage)
    if not prompt_tokens or not written:
        return prompt_tokens
    return max(0, prompt_tokens - written)


def billable_input_tokens(
    prompt_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    prompt_folds_cache: bool,
) -> int:
    """The part of a prompt count that Revenium prices at the input rate.

    ``prompt_tokens`` is the OpenAI and LiteLLM spelling, and whether the cache
    tokens sit inside it in a way this row may remove depends on which upstream
    served the call, which only the caller knows: hence ``prompt_folds_cache``.
    (Anthropic's own ``input_tokens`` spelling reports the cache buckets beside
    it and never reaches this function.)

    Getting that scope wrong costs money in both directions. Left folded for an
    Anthropic response, the cache tokens are priced once at the input rate and
    again in ``cache_read_token_count`` / ``cache_creation_token_count``, which
    is the 4.5x this fix exists for (BACK-3334). Taken out for an OpenAI-shaped
    provider, they are taken out twice: hypercurrent's
    ``AICompletionMetricProcessor.normalizeCacheOverlap`` subtracts
    ``cacheReadTokenCount`` from ``inputTokenCount`` itself for catalog models
    whose provider is OPENAI, GROQ, XAI, AZURE or GEMINI -- its documented
    invariant is that the stored input count is gross -- and clamps a negative
    result to zero, which prices the whole input leg at nothing. A mirror of
    that provider set lives in isotope's ``AiMetricsCosted`` SQL. Anthropic is
    deliberately absent from both, which is why the SDK must remove Claude's
    overlap itself, exactly once per record: here for the LiteLLM rows (client
    and gateway), for direct Anthropic and for Claude on Bedrock or Vertex AI,
    and through ``prompt_folds_claude_cache`` for the OpenAI client's records
    of a gateway serving Claude. Every other model's prompt count is reported
    exactly as LiteLLM built it.

    A count that comes out negative contradicts the cache counts reported
    beside it, so it is reported unchanged and said out loud rather than
    clamped to a figure no provider sent.
    """
    if not prompt_tokens or not prompt_folds_cache:
        return prompt_tokens
    billable = prompt_tokens - cache_read_tokens - cache_creation_tokens
    if billable < 0:
        logger.warning(
            "Usage counts contradict each other: prompt_tokens %s is smaller than "
            "the cache tokens reported inside it (read %s, creation %s); reporting "
            "the prompt count unchanged",
            prompt_tokens, cache_read_tokens, cache_creation_tokens,
        )
        return prompt_tokens
    return billable


ANTHROPIC_MESSAGES_CALL_TYPE = "anthropic_messages"

# LiteLLM ``custom_llm_provider`` values whose usage conversion folds the cache
# buckets into ``prompt_tokens`` *and* whose rows the platform stores as sent.
# Claude's Anthropic-shaped usage is the one usage shape Revenium never
# normalizes server-side (see billable_input_tokens), so it is the only overlap
# this row may remove: here for direct Anthropic, and in
# ``_CLAUDE_HOSTING_PROVIDERS`` for Claude on Bedrock and Vertex AI.
_CACHE_FOLDING_PROVIDERS = frozenset({"anthropic", "anthropic_text"})

# Providers that fold the same way when the model they serve is Claude:
# LiteLLM's Bedrock Converse, invoke and mantle routes and its Vertex AI Claude
# route all build ``prompt_tokens`` as uncached input plus cache read plus cache
# write. Gemini on Vertex AI is reported gross by Google and netted by the
# platform, and Nova on Bedrock is left as LiteLLM built it.
_CLAUDE_HOSTING_PROVIDERS = frozenset({"bedrock", "vertex_ai"})

# Only the top-level fields LiteLLM fills with the deployment it called.
_MODEL_FIELDS = ("model", "litellm_model_name")


def names_claude_model(model: Any) -> bool:
    """Whether a model id names Claude, the way LiteLLM itself tells on Bedrock and Vertex AI."""
    return "claude" in str(model or "").lower()


_OPENAI_CACHE_DETAILS = ("prompt_tokens_details", "input_tokens_details")
_OPENAI_CACHE_DETAIL_COUNTS = ("cached_tokens", "cache_write_tokens")


def _reports_openai_cache_details(usage: Any) -> bool:
    return any(
        _optional_token_count(_get(usage, details_name), count_name) is not None
        for details_name in _OPENAI_CACHE_DETAILS
        for count_name in _OPENAI_CACHE_DETAIL_COUNTS
    )


def prompt_folds_claude_cache(model: Any, usage: Any) -> bool:
    """Whether an OpenAI-shaped prompt count for Claude has the cache inside it.

    An OpenAI-compatible gateway serving Claude (LiteLLM, OpenRouter) folds
    Anthropic's cache read and write into ``prompt_tokens`` and reports them in
    OpenAI's detail fields, and the platform never nets Claude rows (see
    ``billable_input_tokens``). LangChain's Anthropic records carry no OpenAI
    detail fields and arrive already net, so they are left alone.
    """
    return names_claude_model(model) and _reports_openai_cache_details(usage)


def _level(source: Any) -> Any:
    return source if hasattr(source, "get") else {}


def _first_named_model(sources) -> Any:
    for source in sources:
        level = _level(source)
        for field in _MODEL_FIELDS:
            model = level.get(field)
            if model:
                return model
    return None


def _upstream_provider(sources) -> Optional[str]:
    """The lower-cased ``custom_llm_provider`` of the first source naming one, or None.

    An ``anthropic_messages`` call type met first stands for ``anthropic``.
    """
    for source in sources:
        level = _level(source)
        provider = level.get("custom_llm_provider")
        if provider:
            return str(provider).lower()
        if level.get("call_type") == ANTHROPIC_MESSAGES_CALL_TYPE:
            return "anthropic"
    return None


def _provider_folds_cache(provider: str, sources) -> bool:
    if provider in _CACHE_FOLDING_PROVIDERS:
        return True
    return provider in _CLAUDE_HOSTING_PROVIDERS and names_claude_model(_first_named_model(sources))


def prompt_tokens_fold_cache(*sources):
    """True when this call's ``prompt_tokens`` is an Anthropic count plus cache.

    Only the upstream decides, never the numbers: an OpenAI-shaped provider
    reports ``cached_tokens`` inside a prompt count the platform itself nets
    out, so subtracting here would net it out twice and price the input leg at
    zero. See ``billable_input_tokens`` for both failure modes.

    ``custom_llm_provider`` is the field to read, and which surface carries it
    depends on the hook, so every caller hands in the ones it has and the first
    that names a provider decides, in the order the caller passes them. The
    gateway hooks pass the response's own before the request's; the LiteLLM
    client passes the provider it resolved from the request first, because a
    non-streamed response does not name one.
    Only surfaces LiteLLM fills qualify: ``data["metadata"]`` on ``/v1/messages``
    is the caller's request body, so a value there would let a caller pick the
    arithmetic that sets their billed input count (see BACK-3190 for the same
    rule on headers).
    Measured against a live litellm 1.102.0 proxy: the logging event's
    ``kwargs`` and its ``litellm_params`` both carry it (also on the 1.93.0
    floor), while on the non-streamed ``/v1/messages`` route the post-call hook
    is handed a raw provider body with no ``_hidden_params`` and a request dict
    that names no provider -- there it is ``data["litellm_logging_obj"]``'s
    ``model_call_details`` that carries both the provider and the call type.
    An ``anthropic_messages`` call type says the same thing about a route
    LiteLLM only ever serves from an Anthropic-shaped upstream, and covers a
    surface carrying the call type but no provider.

    On ``bedrock`` and ``vertex_ai`` the model decides as well: Claude there is
    folded like Anthropic (BACK-4162), anything else is not. The model is the
    first top-level ``model`` or ``litellm_model_name`` on any source, both
    filled by LiteLLM with the deployment it called. The caller's alias, in the
    request body, its metadata or a gateway's non-streamed ``response.model``,
    is never handed in by a gateway hook, so it cannot turn a Nova
    deployment's count into a Claude one.

    With nothing to read, the prompt count stays as LiteLLM built it: reporting
    a gross count costs the cache overlap twice for Anthropic only, while
    reporting a net one costs the whole input leg for every pool the platform
    normalizes.
    """
    provider = _upstream_provider(sources)
    if provider:
        return _provider_folds_cache(provider, sources)
    logger.debug(
        "No upstream provider on this event; reporting prompt_tokens as LiteLLM "
        "built it rather than assuming the cache tokens are inside it"
    )
    return False


# Release wheels: from 1.66.0 LiteLLM added only the cache read to
# ``prompt_tokens``. ``AnthropicConfig.calculate_usage`` (direct Anthropic, and
# inherited by Vertex AI Claude and Bedrock invoke) added the write again from
# 1.77.3; Bedrock Converse's ``transform_usage`` only from 1.78.0. 1.65 and
# earlier added both on every route.
_ANTHROPIC_STYLE_READ_ONLY_FOLD_RELEASES = ((1, 66, 0), (1, 77, 3))
_CONVERSE_READ_ONLY_FOLD_RELEASES = ((1, 66, 0), (1, 78, 0))

_BEDROCK_CONVERSE_ROUTES = frozenset({"converse", "converse_like"})


def _installed_litellm_version() -> Optional[str]:
    return installed_version("litellm")


def litellm_folds_cache_write(converse: bool) -> bool:
    """Whether the installed LiteLLM counts a cache write inside ``prompt_tokens``.

    ``converse`` says the call went through Bedrock Converse rather than an
    Anthropic-style route. An absent or unreadable version keeps the answer
    every current LiteLLM gives.
    """
    release = release_of(_installed_litellm_version())
    if release is None:
        return True
    first, end = _CONVERSE_READ_ONLY_FOLD_RELEASES if converse else _ANTHROPIC_STYLE_READ_ONLY_FOLD_RELEASES
    return not first <= release < end


def _bedrock_model_routes_via_converse(model: str) -> bool:
    try:
        from litellm.llms.bedrock.common_utils import BedrockModelInfo

        return BedrockModelInfo.get_bedrock_route(model) in _BEDROCK_CONVERSE_ROUTES
    except Exception:
        # LiteLLM routes Claude on Bedrock through Converse unless the model names invoke.
        return "invoke/" not in model


def routes_via_bedrock_converse(*sources) -> bool:
    """Whether LiteLLM served this call through Bedrock Converse.

    Read from the same sources, in the same order, as ``prompt_tokens_fold_cache``,
    and decided by LiteLLM's own Bedrock routing for the deployment's model.
    """
    if _upstream_provider(sources) != "bedrock":
        return False
    return _bedrock_model_routes_via_converse(str(_first_named_model(sources) or ""))


def cache_creation_inside_litellm_prompt(cache_creation_tokens: int, *sources) -> int:
    """The part of a cache write that LiteLLM's ``prompt_tokens`` contains.

    What ``billable_input_tokens`` may take out of a folded LiteLLM prompt
    count; the write itself is still reported in full beside it. ``sources``
    are the ones handed to ``prompt_tokens_fold_cache`` for the same call.
    """
    if litellm_folds_cache_write(converse=routes_via_bedrock_converse(*sources)):
        return cache_creation_tokens
    return 0


def total_priced_tokens(
    input_tokens: int, output_tokens: int, cache_read_tokens: int, cache_creation_tokens: int
) -> int:
    """Every token the platform prices for one call, summed.

    The cache counts are outside ``input_token_count`` once the split above has
    been applied, so a total derived as input + output alone would stop
    matching the total Claude Code's own telemetry reports for the same call --
    the record a gateway row has to agree with.
    """
    return input_tokens + output_tokens + cache_read_tokens + cache_creation_tokens


def _optional_token_count(source: Any, name: str) -> Optional[int]:
    """Read an integral token count, or None when it is absent or unusable.

    Unlike `get_usage_field`, a missing field is reported as None rather than
    coerced to 0, so callers can tell "the provider reported zero" apart from
    "the provider reported nothing". Non-numeric values (including the
    attributes that attribute-style test doubles auto-create on access) count
    as nothing rather than being forwarded into a metering payload.
    """
    value = _get(source, name)
    if isinstance(value, bool) or not isinstance(value, Number):
        return None
    return int(value)


_SPLIT_PARAMETERS = (
    ("ephemeral_5m_input_tokens", "cache_creation5m_token_count"),
    ("ephemeral_1h_input_tokens", "cache_creation1h_token_count"),
)
_CONVERSE_TTL_PARAMETERS = {
    "5m": "cache_creation5m_token_count",
    "1h": "cache_creation1h_token_count",
}


def _reported_split(split: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for provider_name, parameter in _SPLIT_PARAMETERS:
        tokens = _optional_token_count(split, provider_name)
        if tokens is not None:
            counts[parameter] = tokens
    return counts


def _converse_split(usage: Any) -> Dict[str, int]:
    """Bedrock Converse's ``cacheDetails`` list, kept only when it is the whole write.

    The same rule as LiteLLM's own Converse parser (``_parse_cache_details``):
    a list that leaves part of ``cacheWriteInputTokens`` unaccounted for, or
    names a TTL Revenium has no price for, would understate the write.
    """
    details = _get(usage, "cacheDetails")
    if not isinstance(details, list) or not details:
        return {}
    counts = dict.fromkeys(_CONVERSE_TTL_PARAMETERS.values(), 0)
    for detail in details:
        parameter = _CONVERSE_TTL_PARAMETERS.get(_get(detail, "ttl"))
        tokens = _optional_token_count(detail, "inputTokens")
        if parameter is None or tokens is None:
            return {}
        counts[parameter] += tokens
    if sum(counts.values()) != _optional_token_count(usage, "cacheWriteInputTokens"):
        return {}
    return counts


def extract_cache_creation_ttl_counts(usage: Any) -> Dict[str, int]:
    """Extract the per-TTL cache-creation breakdown as metering parameters.

    Three usage shapes carry the split, and this reads whichever is present:

    - Anthropic, natively and on Bedrock InvokeModel: ``usage.cache_creation``
      (``ephemeral_5m_input_tokens`` / ``ephemeral_1h_input_tokens``).
    - LiteLLM's conversion of an Anthropic response: the same two keys under
      ``usage.prompt_tokens_details.cache_creation_token_details``.
    - Bedrock Converse: ``usage.cacheDetails``, one ``{ttl, inputTokens}``
      entry per TTL.

    The split is absent on older responses and whenever the extended cache-TTL
    beta is not in play, so it is read defensively.

    Returns metering parameter names mapped to the counts the provider
    actually reported, ready to be merged into a metering payload. A bucket the
    provider did not report is left out entirely instead of being sent as a
    zero: the aggregate ``cache_creation_token_count`` stays authoritative and
    the backend prices the flat fallback from it when no split is present,
    whereas an explicit zero would assert a split that never happened. A
    bucket the provider does report as 0 is forwarded as 0.
    """
    return (
        _reported_split(_get(usage, "cache_creation"))
        or _reported_split(_get(_get(usage, "prompt_tokens_details"), "cache_creation_token_details"))
        or _converse_split(usage)
    )
