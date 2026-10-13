"""Every public method in a provider resource tree is registered in the manifest or allow-listed with a reason.

Runs on whatever provider versions are installed, so on a provider bump (and in
release-verify, against the versions customers resolve that day) a new public
method that nobody has classified fails here.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest
import yaml

from entry_points import resource_tree
from entry_points.instrumented import selected_environment

HERE = pathlib.Path(__file__).parent
INVENTORY = yaml.safe_load((HERE / "resource_inventory.yaml").read_text())["providers"]
MANIFEST_ROW_IDS = {row["id"] for row in yaml.safe_load((HERE / "manifest.yaml").read_text())["rows"]}
SDKS_IN_THIS_ENVIRONMENT = frozenset(selected_environment().provider_sdks)
IMPLEMENTED_HOOKS_REASON = "LangChain callback hook implemented by UnifiedReveniumCallbackHandler"
# One method per root that the walk must reach, so a root that silently yields
# nothing fails instead of passing with an empty unclassified set.
ANCHORS = {
    "openai": {"OpenAI.chat.completions.create", "OpenAI.chat.completions.with_raw_response.create",
               "AsyncOpenAI.responses.with_streaming_response.create"},
    "anthropic": {"Anthropic.messages.create", "AsyncAnthropic.messages.with_raw_response.create"},
    "perplexity": {"Perplexity.chat.completions.create"},
    "google-genai": {"Client.models.generate_content", "Client.aio.models.generate_content"},
    "ollama": {"Client.chat", "AsyncClient.chat", "ollama.chat"},
    "litellm": {"litellm.completion", "litellm.acreate", "litellm.anthropic_interface.messages.acreate"},
    "fal": {"SyncClient.run", "AsyncClient.run", "SyncRequestHandle.get", "fal_client.run"},
    "vertex": {"generative_models.GenerativeModel.generate_content",
               "generative_models.ChatSession.send_message",
               "preview.generative_models.GenerativeModel.generate_content_async",
               "language_models.TextGenerationModel.predict",
               "language_models.TextEmbeddingModel.get_embeddings",
               "vision_models.ImageCaptioningModel.get_captions",
               "preview.vision_models.ImageCaptioningModel.get_captions",
               "batch_prediction.BatchPredictionJob.submit",
               "tuning.sft.train",
               "preview.caching.CachedContent.create"},
    "bedrock": {"bedrock-runtime.invoke_model", "bedrock-runtime.converse"},
    "langchain": {"AsyncCallbackHandler.on_llm_end"},
    "griptape": {"BasePromptDriver.run", "OpenAiChatPromptDriver.try_run", "OpenAiEmbeddingDriver.try_embed_chunk"},
}


def _registered(provider):
    return set(INVENTORY[provider]["registered"])


def _allowed(provider):
    return {method for group in INVENTORY[provider]["allowed"] for method in group["methods"]}


def _mirror_base(method):
    for mirror in resource_tree.RAW_RESPONSE_MIRRORS:
        segment = f".{mirror}."
        if segment in method:
            return method.replace(segment, ".", 1)
    return None


def _classified(method, registered, allowed):
    if method in registered or method in allowed:
        return True
    # A raw or streaming-response form reaches the same endpoint as its base, so an
    # allowed base covers it. A registered base does not: its raw form is a separate
    # code path through the wrap, and must be named with the rows that execute it.
    return _mirror_base(method) in allowed


def unclassified(provider):
    registered, allowed = _registered(provider), _allowed(provider)
    return {method for method in resource_tree.public_methods(provider)
            if not _classified(method, registered, allowed)}


def _installed(provider):
    package, _roots = resource_tree.PROVIDERS[provider]
    if provider in resource_tree.PACKAGES_FOUND_BY_IMPORT:
        return importlib.util.find_spec(package) is not None
    return package in SDKS_IN_THIS_ENVIRONMENT


def _provider_param(provider):
    marks = [] if _installed(provider) else [pytest.mark.skip(reason=f"{provider} is not installed in this environment")]
    return pytest.param(provider, id=provider, marks=marks)


PROVIDER_PARAMS = [_provider_param(provider) for provider in resource_tree.PROVIDERS]


def _require(provider):
    if not _installed(provider):
        pytest.skip(f"{provider} is not installed in this environment")


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_every_public_method_is_registered_or_allowed(provider):
    missing = sorted(unclassified(provider))
    assert missing == [], (
        f"{len(missing)} public {provider} method(s) are neither registered nor allow-listed in "
        f"tests/entry_points/resource_inventory.yaml. Add each one to `registered` with the manifest rows "
        f"that execute it, or to an `allowed` group whose reason fits: {missing}"
    )


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_the_walk_reaches_the_registered_entry_points(provider):
    if not _registered(provider):
        pytest.skip(f"{provider} has no registered methods; no manifest row executes it directly")
    assert resource_tree.public_methods(provider) & _registered(provider)


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_the_walk_reaches_every_root(provider):
    assert ANCHORS[provider] - resource_tree.public_methods(provider) == set()


def test_every_wrapped_sdk_and_framework_has_anchors():
    assert set(ANCHORS) == set(resource_tree.PROVIDERS)


def test_a_new_public_method_fails_the_inventory(monkeypatch):
    _require("openai")
    import openai.resources.embeddings as embeddings
    monkeypatch.setattr(embeddings.Embeddings, "brand_new_endpoint", lambda self: None, raising=False)
    assert unclassified("openai") == {"OpenAI.embeddings.brand_new_endpoint"}


def test_a_new_method_on_the_client_object_fails_the_inventory(monkeypatch):
    _require("openai")
    import openai
    monkeypatch.setattr(openai.OpenAI, "brand_new_root_call", lambda self: None, raising=False)
    assert unclassified("openai") == {"OpenAI.brand_new_root_call"}


def test_a_subtree_deeper_than_the_walk_limit_fails_the_inventory(monkeypatch):
    _require("openai")
    monkeypatch.setattr(resource_tree, "MAX_DEPTH", 1)
    assert "OpenAI.chat.completions.<truncated>" in unclassified("openai")


def test_a_new_ollama_method_fails_the_inventory(monkeypatch):
    _require("ollama")
    import ollama
    monkeypatch.setattr(ollama.Client, "brand_new_call", lambda self: None, raising=False)
    assert unclassified("ollama") == {"Client.brand_new_call"}


def test_a_new_langchain_callback_hook_fails_the_inventory(monkeypatch):
    _require("langchain")
    from langchain_core.callbacks.base import AsyncCallbackHandler
    monkeypatch.setattr(AsyncCallbackHandler, "on_brand_new_event", lambda self: None, raising=False)
    assert unclassified("langchain") == {"AsyncCallbackHandler.on_brand_new_event"}


def test_a_raw_form_of_a_metered_method_must_be_classified(monkeypatch):
    _require("openai")
    raw_create = "OpenAI.chat.completions.with_raw_response.create"
    monkeypatch.delitem(INVENTORY["openai"]["registered"], raw_create, raising=False)
    assert raw_create in unclassified("openai")


def test_a_raw_form_inherits_an_allowed_base():
    _require("openai")
    raw_images = "OpenAI.images.with_raw_response.generate"
    assert raw_images in resource_tree.public_methods("openai")
    assert raw_images not in unclassified("openai")


def test_the_litellm_walk_does_not_depend_on_which_names_were_touched_first():
    _require("litellm")
    from litellm._lazy_imports import _get_lazy_import_registry
    import litellm
    for name in _get_lazy_import_registry():
        getattr(litellm, name, None)
    fresh = subprocess.run(
        [sys.executable, "-c",
         "import json; from entry_points import resource_tree; "
         "print(json.dumps(sorted(resource_tree.public_methods('litellm'))))"],
        cwd=HERE.parent, capture_output=True, text=True, check=True,
    )
    assert set(json.loads(fresh.stdout.splitlines()[-1])) == resource_tree.public_methods("litellm")


def test_hooks_the_inventory_says_we_implement_are_implemented():
    _require("langchain")
    from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler
    implemented = {method.rpartition(".")[2]
                   for group in INVENTORY["langchain"]["allowed"]
                   if group["reason"].startswith(IMPLEMENTED_HOOKS_REASON)
                   for method in group["methods"]}
    assert implemented
    assert implemented - set(vars(UnifiedReveniumCallbackHandler)) == set()


class TestInventoryFile:
    def test_every_provider_with_a_resource_tree_has_an_inventory(self):
        assert set(INVENTORY) == set(resource_tree.PROVIDERS)

    def test_registered_methods_name_existing_manifest_rows(self):
        for provider, entry in INVENTORY.items():
            for method, row_ids in entry["registered"].items():
                assert row_ids and set(row_ids) <= MANIFEST_ROW_IDS, (provider, method, row_ids)

    def test_every_allowed_group_has_a_reason_and_methods(self):
        for provider, entry in INVENTORY.items():
            for group in entry["allowed"]:
                assert group["reason"].strip() and group["methods"], (provider, group)

    def test_no_method_is_both_registered_and_allowed(self):
        for provider in INVENTORY:
            assert _registered(provider) & _allowed(provider) == set(), provider

    def test_no_method_is_allowed_twice(self):
        for provider, entry in INVENTORY.items():
            methods = [method for group in entry["allowed"] for method in group["methods"]]
            assert len(methods) == len(set(methods)), provider
