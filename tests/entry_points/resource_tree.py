"""The public call surface of every provider SDK and framework the middleware wraps.

Walking a constructed client, rather than the provider's module layout, names each
method the way a caller reaches it (``AsyncOpenAI.chat.completions.create``), so a
provider that moves a class between modules does not change the inventory. A root
may also be a module (``ollama.chat``, ``litellm.completion``) or a class the
middleware wraps or subclasses (``generative_models.GenerativeModel.generate_content``).
Vertex AI also has its batch prediction, supervised tuning and context-cache entry
points walked, and boto3's Bedrock Runtime client is walked as a constructed client.

CrewAI is not walked: no extra or CI environment installs it, and its model calls
go through ``litellm.completion``, which the litellm root covers.
"""
import functools
import importlib
import inspect
import warnings

RAW_RESPONSE_MIRRORS = frozenset({"with_raw_response", "with_streaming_response"})
QUERY_STRING_SERIALIZER = "qs"
# Deepest accessor chain on 2026-10-05: six hops on anthropic 1.11
# (Anthropic.beta.organization.analytics.apps.chat.projects), five on openai 2.54,
# three on perplexity and google-genai. Module and class roots are not descended.
# Ten leaves four hops of headroom; a deeper subtree is recorded as TRUNCATED,
# which the inventory test fails on.
MAX_DEPTH = 10
TRUNCATED = "<truncated>"
FRAMEWORKS = frozenset({"langchain", "griptape"})
# Walked wherever their package imports, whichever matrix environment is selected:
# no instrumented.py environment lists them, since that list drives the patch checks.
PACKAGES_FOUND_BY_IMPORT = FRAMEWORKS | {"bedrock"}
UNREACHABLE_HOST = "http://127.0.0.1:1"
VERTEX_MODEL_MODULES = (
    "vertexai.generative_models",
    "vertexai.preview.generative_models",
    "vertexai.language_models",
    "vertexai.vision_models",
    "vertexai.preview.vision_models",
)
VERTEX_ROOT_SUFFIXES = ("Model", "Session")
VERTEX_JOB_AND_CACHE_CLASSES = (
    "vertexai.batch_prediction.BatchPredictionJob",
    "vertexai.preview.caching.CachedContent",
)
VERTEX_JOB_MODULES = ("vertexai.tuning.sft",)
BEDROCK_REGION = "us-east-1"
GRIPTAPE_DRIVERS = (
    "griptape.drivers.prompt.base_prompt_driver.BasePromptDriver",
    "griptape.drivers.embedding.base_embedding_driver.BaseEmbeddingDriver",
    "griptape.drivers.prompt.openai_chat_prompt_driver.OpenAiChatPromptDriver",
    "griptape.drivers.prompt.azure_openai_chat_prompt_driver.AzureOpenAiChatPromptDriver",
    "griptape.drivers.prompt.anthropic.AnthropicPromptDriver",
    "griptape.drivers.prompt.ollama.OllamaPromptDriver",
    "griptape.drivers.embedding.openai.OpenAiEmbeddingDriver",
)


class LazyModule:
    """A module root whose namespace is partly loaded on first attribute access."""

    def __init__(self, module, lazy_names):
        self.module = module
        self.lazy_names = frozenset(lazy_names)


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


def _ollama_roots():
    import ollama
    return {"Client": ollama.Client(host=UNREACHABLE_HOST),
            "AsyncClient": ollama.AsyncClient(host=UNREACHABLE_HOST),
            "ollama": ollama}


def _litellm_lazy_names():
    try:
        from litellm._lazy_imports import _get_lazy_import_registry
    except ImportError:
        return ()
    return _get_lazy_import_registry()


def _litellm_roots():
    import litellm
    import litellm.anthropic_interface
    return {"litellm": LazyModule(litellm, _litellm_lazy_names()),
            "litellm.anthropic_interface.messages": litellm.anthropic_interface.messages}


def _fal_roots():
    import fal_client
    import fal_client.client
    return {"SyncClient": fal_client.SyncClient(key="stub"),
            "AsyncClient": fal_client.AsyncClient(key="stub"),
            "SyncRequestHandle": fal_client.client.SyncRequestHandle,
            "AsyncRequestHandle": fal_client.client.AsyncRequestHandle,
            "fal_client": fal_client}


def _vertex_roots():
    roots = {}
    for module_name in VERTEX_MODEL_MODULES:
        module = importlib.import_module(module_name)
        prefix = module_name[len("vertexai."):]
        for name, value in vars(module).items():
            if (isinstance(value, type) and not name.startswith("_") and name.endswith(VERTEX_ROOT_SUFFIXES)
                    and value.__module__.startswith("vertexai")):
                roots[f"{prefix}.{name}"] = value
    for dotted_path in VERTEX_JOB_AND_CACHE_CLASSES:
        roots[dotted_path[len("vertexai."):]] = _import_class(dotted_path)
    for module_name in VERTEX_JOB_MODULES:
        roots[module_name[len("vertexai."):]] = importlib.import_module(module_name)
    # google-cloud-aiplatform 1.x ships vertexai.caching without an __init__, so
    # the package imports as an empty namespace; only 2.x exports the class there.
    try:
        from vertexai.caching import CachedContent
    except ImportError:
        pass
    else:
        roots["caching.CachedContent"] = CachedContent
    return roots


def _bedrock_roots():
    import boto3
    return {"bedrock-runtime": boto3.client(
        "bedrock-runtime", region_name=BEDROCK_REGION,
        aws_access_key_id="testing", aws_secret_access_key="testing")}


def _langchain_roots():
    from langchain_core.callbacks.base import AsyncCallbackHandler
    return {"AsyncCallbackHandler": AsyncCallbackHandler}


def _import_class(dotted_path):
    module_name, _, class_name = dotted_path.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def _griptape_roots():
    return {path.rpartition(".")[2]: _import_class(path) for path in GRIPTAPE_DRIVERS}


PROVIDERS = {
    "openai": ("openai", _openai_roots),
    "anthropic": ("anthropic", _anthropic_roots),
    "perplexity": ("perplexity", _perplexity_roots),
    "google-genai": ("google.genai", _genai_roots),
    "ollama": ("ollama", _ollama_roots),
    "litellm": ("litellm", _litellm_roots),
    "fal": ("fal_client", _fal_roots),
    "vertex": ("vertexai", _vertex_roots),
    "bedrock": ("boto3", _bedrock_roots),
    "langchain": ("langchain_core", _langchain_roots),
    "griptape": ("griptape", _griptape_roots),
}


def _is_accessor(static):
    return isinstance(static, (property, functools.cached_property))


def _is_public(name):
    return not name.startswith("_")


def _public_names(names):
    return sorted(name for name in names if _is_public(name) and name != QUERY_STRING_SERIALIZER)


def _is_method(value):
    return callable(value) and not isinstance(value, type)


def _is_resource(value, package):
    cls = type(value)
    return (not isinstance(value, type)
            and cls.__module__.startswith(package)
            and not any(base.__name__ == "BaseModel" for base in cls.__mro__))


def _add_mirror_methods(obj, path, mirror_name, found):
    mirror = getattr(obj, mirror_name)
    for name in _public_names(vars(mirror)):
        if _is_method(vars(mirror)[name]):
            found.add(f"{path}.{mirror_name}.{name}")


def _walk(obj, path, package, depth, found):
    cls = type(obj)
    for name in _public_names(dir(cls)):
        static = inspect.getattr_static(cls, name, None)
        if name in RAW_RESPONSE_MIRRORS and _is_accessor(static):
            _add_mirror_methods(obj, path, name, found)
        elif _is_accessor(static):
            child = getattr(obj, name)
            if not _is_resource(child, package):
                continue
            if depth >= MAX_DEPTH:
                found.add(f"{path}.{name}.{TRUNCATED}")
            else:
                _walk(child, f"{path}.{name}", package, depth + 1, found)
        elif _is_method(static):
            found.add(f"{path}.{name}")


def _walk_module(module, path, package, found, lazy_names=()):
    for name in _public_names(set(dir(module)) | set(lazy_names)):
        try:
            value = getattr(module, name)
        except (AttributeError, ImportError):
            continue
        if _is_method(value) and (getattr(value, "__module__", None) or "").startswith(package):
            found.add(f"{path}.{name}")


def _walk_class(cls, path, found):
    for name in _public_names(dir(cls)):
        static = inspect.getattr_static(cls, name, None)
        if isinstance(static, (staticmethod, classmethod)) or (_is_method(static) and not _is_accessor(static)):
            found.add(f"{path}.{name}")


def _walk_root(root, path, package, found):
    if isinstance(root, LazyModule):
        _walk_module(root.module, path, package, found, root.lazy_names)
    elif inspect.ismodule(root):
        _walk_module(root, path, package, found)
    elif isinstance(root, type):
        _walk_class(root, path, found)
    else:
        _walk(root, path, package, 0, found)


def public_methods(provider):
    """Every public method reachable from the provider's roots, as dotted caller paths."""
    package, roots = PROVIDERS[provider]
    found = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for root_name, root in roots().items():
            _walk_root(root, root_name, package, found)
    return found
