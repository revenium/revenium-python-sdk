"""Tolerant field access for provider usage objects (dict or attribute-style).

Cache token fields differ by provider surface:

- Anthropic-style: top-level ``cache_read_input_tokens`` / ``cache_creation_input_tokens``,
  plus the nested per-TTL breakdown under ``cache_creation``.
- OpenAI-style: nested ``prompt_tokens_details.cached_tokens``; no separate
  cache-creation count.

See BACK-2391 for the history of this bug class (hardcoded/zeroed cache-token
fields recurring across multiple emitters) and BACK-1925 for the LiteLLM
client's cache-token contract, which this module's defaults must not break.
"""
from numbers import Number
from typing import Any, Dict, NamedTuple, Optional


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
    """
    prompt_details = _get(usage, "prompt_tokens_details")
    cache_read_tokens = get_usage_field(prompt_details, "cached_tokens", 0)
    if not cache_read_tokens:
        cache_read_tokens = get_usage_field(usage, "cache_read_input_tokens", 0)
    cache_creation_tokens = get_usage_field(usage, "cache_creation_input_tokens", 0)

    return CacheTokens(cache_read_tokens=cache_read_tokens, cache_creation_tokens=cache_creation_tokens)


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


def extract_cache_creation_ttl_counts(usage: Any) -> Dict[str, int]:
    """Extract the per-TTL cache-creation breakdown as metering parameters.

    Anthropic reports the split under a nested ``usage.cache_creation`` object
    (``ephemeral_5m_input_tokens`` / ``ephemeral_1h_input_tokens``). That
    object is absent on older responses and whenever the extended cache-TTL
    beta is not in play, so it is read defensively.

    Returns metering parameter names mapped to the counts the provider
    actually reported, ready to be merged into a metering payload. A bucket the
    provider did not report is left out entirely instead of being sent as a
    zero: the aggregate ``cache_creation_token_count`` stays authoritative and
    the backend prices the flat fallback from it when no split is present,
    whereas an explicit zero would assert a split that never happened. A
    bucket the provider does report as 0 is forwarded as 0.
    """
    cache_creation = _get(usage, "cache_creation")

    counts: Dict[str, int] = {}
    ephemeral_5m = _optional_token_count(cache_creation, "ephemeral_5m_input_tokens")
    if ephemeral_5m is not None:
        counts["cache_creation5m_token_count"] = ephemeral_5m
    ephemeral_1h = _optional_token_count(cache_creation, "ephemeral_1h_input_tokens")
    if ephemeral_1h is not None:
        counts["cache_creation1h_token_count"] = ephemeral_1h
    return counts
