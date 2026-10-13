"""The short names xAI publishes for each Grok model, keyed by the model id
xAI answers with and meters. Copied whole from xAI's
``GET https://api.x.ai/v1/language-models`` on 2026-10-11; a name xAI adds
or repoints later is known only after this table is refreshed."""
from types import MappingProxyType
from typing import Dict, FrozenSet, Mapping, Set, Tuple

XAI_MODEL_ALIASES: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "grok-4.20-0309-non-reasoning": (
        "grok-4.20-non-reasoning",
        "grok-4.20-non-reasoning-latest",
        "grok-4.20-beta-non-reasoning",
        "grok-4.20-beta-latest-non-reasoning",
        "grok-4.20-experimental-beta-0304-non-reasoning",
        "grok-4.20-experimental-beta-non-reasoning-latest",
        "grok-4.20-beta-0309-non-reasoning",
        "grok-4.20-non-reasoning-gv2",
    ),
    "grok-4.20-0309-reasoning": (
        "grok-4.20-reasoning-latest",
        "grok-4.20",
        "grok-4.20-reasoning",
        "grok-4.20-0309",
        "grok-4.20-beta-0309-reasoning",
        "grok-4.20-beta",
        "grok-4.20-beta-0309",
        "grok-4.20-beta-latest",
        "grok-4.20-beta-latest-reasoning",
        "grok-4.20-beta-reasoning",
        "grok-4.20-experimental-beta-0304-reasoning",
        "grok-4.20-experimental-beta-0304",
        "grok-4.20-experimental-beta-reasoning-latest",
        "grok-4.20-experimental-beta-latest",
        "grok-4.20-reasoning-gv2",
    ),
    "grok-4.20-multi-agent-0309": (
        "grok-4.20-multi-agent",
        "grok-4.20-multi-agent-latest",
        "grok-4.20-multi-agent-beta-latest",
        "grok-4.20-multi-agent-experimental-beta-0304",
        "grok-4.20-multi-agent-experimental-beta-latest",
        "grok-4.20-multi-agent-beta-0309",
    ),
    "grok-4.3": (
        "grok-4.3-latest",
    ),
    "grok-4.5": (
        "grok-4.5-latest",
        "grok-build-latest",
    ),
    "grok-4.6": (),
    "grok-4.7": (),
    "grok-build-0.1": (
        "grok-code-fast-1",
        "grok-code-fast",
        "grok-code-fast-1-0825",
    ),
})


def _served_models_by_alias(aliases_by_served: Mapping[str, Tuple[str, ...]]) -> Mapping[str, FrozenSet[str]]:
    served_by_alias: Dict[str, Set[str]] = {}
    for served, aliases in aliases_by_served.items():
        for alias in aliases:
            served_by_alias.setdefault(alias, set()).add(served)
    return MappingProxyType({alias: frozenset(served) for alias, served in served_by_alias.items()})


SERVED_MODEL_BY_ALIAS: Mapping[str, FrozenSet[str]] = _served_models_by_alias(XAI_MODEL_ALIASES)
