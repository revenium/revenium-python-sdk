"""Every public method in a provider resource tree is registered in the manifest or allow-listed with a reason.

Runs on whatever provider versions are installed, so on a provider bump (and in
release-verify, against the versions customers resolve that day) a new public
method that nobody has classified fails here.
"""
import pathlib

import pytest
import yaml

from entry_points import resource_tree
from entry_points.instrumented import selected_environment

HERE = pathlib.Path(__file__).parent
INVENTORY = yaml.safe_load((HERE / "resource_inventory.yaml").read_text())["providers"]
MANIFEST_ROW_IDS = {row["id"] for row in yaml.safe_load((HERE / "manifest.yaml").read_text())["rows"]}
SDKS_IN_THIS_ENVIRONMENT = frozenset(selected_environment().provider_sdks)


def _registered(provider):
    return set(INVENTORY[provider]["registered"])


def _allowed(provider):
    return {method for group in INVENTORY[provider]["allowed"] for method in group["methods"]}


def unclassified(provider):
    return resource_tree.public_methods(provider) - _registered(provider) - _allowed(provider)


def _installed(provider):
    package, _roots = resource_tree.PROVIDERS[provider]
    return package in SDKS_IN_THIS_ENVIRONMENT


def _provider_param(provider):
    marks = [] if _installed(provider) else [pytest.mark.skip(reason=f"{provider} is not installed in this environment")]
    return pytest.param(provider, id=provider, marks=marks)


PROVIDER_PARAMS = [_provider_param(provider) for provider in resource_tree.PROVIDERS]


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
    assert resource_tree.public_methods(provider) & _registered(provider)


def test_a_new_public_method_fails_the_inventory(monkeypatch):
    if not _installed("openai"):
        pytest.skip("openai is not installed in this environment")
    import openai.resources.embeddings as embeddings
    monkeypatch.setattr(embeddings.Embeddings, "brand_new_endpoint", lambda self: None, raising=False)
    assert unclassified("openai") == {"OpenAI.embeddings.brand_new_endpoint"}


def test_a_new_method_on_the_client_object_fails_the_inventory(monkeypatch):
    if not _installed("openai"):
        pytest.skip("openai is not installed in this environment")
    import openai
    monkeypatch.setattr(openai.OpenAI, "brand_new_root_call", lambda self: None, raising=False)
    assert unclassified("openai") == {"OpenAI.brand_new_root_call"}


def test_a_subtree_deeper_than_the_walk_limit_fails_the_inventory(monkeypatch):
    if not _installed("openai"):
        pytest.skip("openai is not installed in this environment")
    monkeypatch.setattr(resource_tree, "MAX_DEPTH", 1)
    assert "OpenAI.chat.completions.<truncated>" in unclassified("openai")


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
