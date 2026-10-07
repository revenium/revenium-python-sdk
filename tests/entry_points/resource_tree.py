"""The public resource tree of the provider SDKs that have one, read from live client objects.

Walking a constructed client, rather than the provider's module layout, names each
method the way a caller reaches it (``AsyncOpenAI.chat.completions.create``), so a
provider that moves a class between modules does not change the inventory.
"""
import functools
import inspect
import warnings

RAW_RESPONSE_MIRRORS = frozenset({"with_raw_response", "with_streaming_response"})
QUERY_STRING_SERIALIZER = "qs"
SKIPPED_ATTRIBUTES = RAW_RESPONSE_MIRRORS | {QUERY_STRING_SERIALIZER}
# Deepest accessor chain on 2026-10-05: six hops on anthropic 1.11
# (Anthropic.beta.organization.analytics.apps.chat.projects), five on openai 2.54,
# three on perplexity and google-genai. Ten leaves four hops of headroom; a deeper
# subtree is recorded as TRUNCATED, which the inventory test fails on.
MAX_DEPTH = 10
TRUNCATED = "<truncated>"


def _openai_roots():
    import openai
    return {"OpenAI": openai.OpenAI(api_key="stub"), "AsyncOpenAI": openai.AsyncOpenAI(api_key="stub")}


def _anthropic_roots():
    import anthropic
    return {"Anthropic": anthropic.Anthropic(api_key="stub"),
            "AsyncAnthropic": anthropic.AsyncAnthropic(api_key="stub")}


def _perplexity_roots():
    import perplexity
    return {"Perplexity": perplexity.Perplexity(api_key="stub"),
            "AsyncPerplexity": perplexity.AsyncPerplexity(api_key="stub")}


def _genai_roots():
    from google import genai
    client = genai.Client(api_key="stub")
    return {"Client": client,
            "Client.chats.create()": client.chats.create(model="stub"),
            "Client.aio.chats.create()": client.aio.chats.create(model="stub")}


PROVIDERS = {
    "openai": ("openai", _openai_roots),
    "anthropic": ("anthropic", _anthropic_roots),
    "perplexity": ("perplexity", _perplexity_roots),
    "google-genai": ("google.genai", _genai_roots),
}


def _is_accessor(static):
    return isinstance(static, (property, functools.cached_property))


def _public_names(cls):
    return sorted(name for name in dir(cls) if not name.startswith("_") and name not in SKIPPED_ATTRIBUTES)


def _is_resource(value, package):
    cls = type(value)
    return (not isinstance(value, type)
            and cls.__module__.startswith(package)
            and not any(base.__name__ == "BaseModel" for base in cls.__mro__))


def _walk(obj, path, package, depth, found):
    cls = type(obj)
    for name in _public_names(cls):
        static = inspect.getattr_static(cls, name, None)
        if _is_accessor(static):
            child = getattr(obj, name)
            if not _is_resource(child, package):
                continue
            if depth >= MAX_DEPTH:
                found.add(f"{path}.{name}.{TRUNCATED}")
            else:
                _walk(child, f"{path}.{name}", package, depth + 1, found)
        elif callable(static) and not isinstance(static, type):
            found.add(f"{path}.{name}")


def public_methods(provider):
    """Every public method reachable from the provider's client objects, as dotted caller paths."""
    package, roots = PROVIDERS[provider]
    found = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for root_name, root in roots().items():
            _walk(root, root_name, package, 0, found)
    return found
