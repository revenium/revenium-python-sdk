"""A Perplexity wrap is reported as installed only when it took effect (BACK-3607)."""
import json
import logging
import subprocess
import sys
import textwrap
import types

import pytest
import wrapt

from revenium_middleware._core.patch_registry import is_patched, unregister_patch
from revenium_middleware.perplexity import perplexity_sdk
from revenium_middleware.perplexity.patching import wrap_registered

FAKE_MODULE = "revenium_test_fake_perplexity_completions"


def _passthrough(wrapped, instance, args, kwargs):
    return wrapped(*args, **kwargs)


@pytest.fixture
def fake_module(monkeypatch):
    module = types.ModuleType(FAKE_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_MODULE, module)
    monkeypatch.setattr(perplexity_sdk, "NATIVE_COMPLETIONS_MODULE", FAKE_MODULE)
    yield module
    for name in ("CompletionsResource", "AsyncCompletionsResource", "Missing"):
        unregister_patch(f"{FAKE_MODULE}.{name}.create")


def _resource(name):
    return type(name, (), {"create": lambda self: name})


class TestWrapRegistered:
    def test_a_wrap_at_a_missing_attribute_is_not_reported_as_installed(self, fake_module, caplog):
        key = f"{FAKE_MODULE}.Missing.create"

        with caplog.at_level(logging.WARNING, logger="revenium_middleware.perplexity"):
            applied = wrap_registered(key, FAKE_MODULE, "Missing.create", _passthrough)

        assert applied is False
        assert not is_patched(key)
        assert [r.levelno for r in caplog.records] == [logging.WARNING]
        assert key in caplog.records[0].getMessage()

    def test_a_wrap_of_an_uninstalled_library_is_not_reported_and_not_warned(self, caplog):
        key = "perplexity:revenium_test_not_installed.Completions.create"

        with caplog.at_level(logging.WARNING, logger="revenium_middleware.perplexity"):
            applied = wrap_registered(key, "revenium_test_not_installed", "Completions.create", _passthrough)

        assert applied is False
        assert not is_patched(key)
        assert caplog.records == []

    def test_a_wrap_that_applies_is_reported_as_installed(self, fake_module):
        fake_module.CompletionsResource = _resource("CompletionsResource")
        key = f"{FAKE_MODULE}.CompletionsResource.create"

        assert wrap_registered(key, FAKE_MODULE, "CompletionsResource.create", _passthrough)
        assert is_patched(key)
        assert isinstance(vars(fake_module.CompletionsResource)["create"], wrapt.FunctionWrapper)


class TestNativeClientTargets:
    def test_the_installed_library_is_wrapped_at_the_classes_it_ships(self):
        assert is_patched("perplexity.resources.chat.completions.CompletionsResource.create")
        assert is_patched("perplexity.resources.chat.completions.AsyncCompletionsResource.create")
        assert not is_patched("perplexity.resources.chat.completions.Completions.create")

    def test_a_release_without_the_resource_classes_is_not_reported_and_is_warned(self, fake_module, caplog):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware.perplexity"):
            perplexity_sdk.patch_native_client()

        assert not is_patched(f"{FAKE_MODULE}.CompletionsResource.create")
        assert not is_patched(f"{FAKE_MODULE}.AsyncCompletionsResource.create")
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(f"{FAKE_MODULE}.CompletionsResource.create" in m for m in messages)
        assert any(f"{FAKE_MODULE}.AsyncCompletionsResource.create" in m for m in messages)


def test_the_native_client_is_wrapped_when_the_openai_package_is_not_installed():
    """The perplexity-native extra does not install openai."""
    script = textwrap.dedent("""
        import json, sys
        sys.modules["openai"] = None
        import revenium_middleware.perplexity as middleware
        from revenium_middleware._core.patch_registry import _patched
        print(json.dumps({"native": middleware.perplexity_create_wrapper is not None,
                          "patched": sorted(_patched)}))
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
                               env={"REVENIUM_METERING_API_KEY": "hak_test_native_only", "PATH": ""})
    assert completed.returncode == 0, completed.stderr[-2000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["native"] is True
    assert result["patched"] == [
        "perplexity.resources.chat.completions.AsyncCompletionsResource.create",
        "perplexity.resources.chat.completions.CompletionsResource.create",
    ]
