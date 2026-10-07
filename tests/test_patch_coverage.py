"""Structural metering coverage, read from the running program rather than source text.

The patch registry records a target before the wrap is attempted, so a registry
entry proves nothing on its own. These tests resolve every registered target to
its live attribute and require a wrapt wrapper owned by our middleware, require
the async twin of every wrapped sync entry point, and require the class-level
method behind every wrapped default-client bound method.

Every allow-list entry names the ticket that removes it. An entry that no longer
suppresses a failure is itself a failure, so a fix forces its entry out.
"""
import importlib
import inspect
import re
import sys
import types

import pytest
import wrapt

from entry_points.instrumented import import_all_middleware, selected_environment, selected_environment_name

import_all_middleware()

from revenium_middleware._core import patch_registry  # noqa: E402

WRAPPER_TYPES = (wrapt.FunctionWrapper, wrapt.BoundFunctionWrapper)
OUR_PACKAGE = "revenium_middleware"

OWNER_BY_ROOT = {
    "openai": "revenium_middleware.openai",
    "anthropic": "revenium_middleware.anthropic",
    "botocore": "revenium_middleware.anthropic",
    "google.genai": "revenium_middleware.google",
    "vertexai": "revenium_middleware.google",
    "ollama": "revenium_middleware.ollama",
    "litellm": "revenium_middleware.litellm",
    "fal_client": "revenium_middleware.fal",
    "perplexity": "revenium_middleware.perplexity",
}

SDKS_IN_THIS_ENVIRONMENT = frozenset(selected_environment().provider_sdks)

UNRESOLVED_ALLOWED = {}

TWIN_ALLOWED = {}

UNRESOLVED_TWIN_ALLOWED = {}

CLASS_LEVEL_ALLOWED = {}

TICKET_KEY = re.compile(r"^BACK-\d+$")


# --- Reading the live program -----------------------------------------------------

def split_key(key):
    """A registry key is an import path, optionally prefixed by an owning namespace (``perplexity:``)."""
    namespace, _, path = key.rpartition(":")
    return namespace, path


def root_of(path):
    for root in sorted(OWNER_BY_ROOT, key=len, reverse=True):
        if path == root or path.startswith(root + "."):
            return root
    return None


def in_this_environment(key):
    """Whether the provider behind ``key`` is installed in the selected test environment."""
    return root_of(split_key(key)[1]) in SDKS_IN_THIS_ENVIRONMENT


def expected_owner(key):
    namespace, path = split_key(key)
    if namespace:
        return f"{OUR_PACKAGE}.{namespace}"
    return OWNER_BY_ROOT.get(root_of(path))


def resolve(path):
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for name in parts[split:]:
                obj = getattr(obj, name)
        except AttributeError as exc:
            raise LookupError(f"{path}: {exc}") from exc
        return obj
    raise LookupError(f"{path}: no importable module prefix")


def resolves(path):
    try:
        resolve(path)
    except LookupError:
        return False
    return True


def wrapper_chain(obj):
    """The wrapt layers around ``obj``, outermost first, and the object they finally wrap."""
    layers = []
    while isinstance(obj, WRAPPER_TYPES):
        layers.append(obj)
        obj = obj.__wrapped__
    return layers, obj


def wrapper_owners(obj):
    layers, _ = wrapper_chain(obj)
    return [layer._self_wrapper.__module__ for layer in layers]


def is_ours(obj, owner=OUR_PACKAGE):
    return any(module == owner or module.startswith(owner + ".") for module in wrapper_owners(obj))


def registered_keys():
    return sorted(patch_registry._patched)


# --- Async twins ---------------------------------------------------------------------

def _async_class_twin(path):
    module_and_class, _, method = path.rpartition(".")
    module, _, cls = module_and_class.rpartition(".")
    if not cls[:1].isupper() or cls.startswith("Async"):
        return None
    return f"{module}.Async{cls}.{method}"


def _litellm_twin(path):
    name = path.rpartition(".")[2]
    return None if name.startswith("a") else f"litellm.a{name}"


def _suffix_async_twin(path):
    return None if path.endswith("_async") else f"{path}_async"


VERTEX_CLASSES_WITHOUT_ASYNC = ("ImageGenerationModel",)


def _has_no_async_by_design(path):
    return root_of(path) == "vertexai" and path.rpartition(".")[0].rpartition(".")[2] in VERTEX_CLASSES_WITHOUT_ASYNC


def _vertex_twin(path):
    return None if _has_no_async_by_design(path) else _suffix_async_twin(path)


def _fal_twin(path):
    module_and_class, _, method = path.rpartition(".")
    module, _, cls = module_and_class.rpartition(".")
    if cls.startswith("Sync"):
        return f"{module}.Async{cls[len('Sync'):]}.{method}"
    if cls.startswith("Async"):
        return None
    return _suffix_async_twin(path)


def _ollama_twin(path):
    return f"ollama.AsyncClient.{path.rpartition('.')[2]}"


TWIN_RULES = {
    "openai": _async_class_twin,
    "anthropic": _async_class_twin,
    "perplexity": _async_class_twin,
    "google.genai": _async_class_twin,
    "litellm": _litellm_twin,
    "fal_client": _fal_twin,
    "ollama": _ollama_twin,
    "vertexai": _vertex_twin,
    "botocore": lambda path: None,
}


def twin_of(key):
    """The registry key the async twin of ``key`` would have, or None when the provider has no twin by design."""
    namespace, path = split_key(key)
    twin_path = TWIN_RULES[root_of(path)](path)
    if twin_path is None:
        return None
    return f"{namespace}:{twin_path}" if namespace else twin_path


def _twins_of_resolvable_sync_entries(keys):
    for key in keys:
        twin = twin_of(key)
        if twin is not None and resolves(split_key(key)[1]):
            yield twin, key


def missing_twins(keys):
    """Twins the provider defines that are not registered, mapped to their sync entry."""
    registered = set(keys)
    return {
        twin: key for twin, key in _twins_of_resolvable_sync_entries(keys)
        if twin not in registered and resolves(split_key(twin)[1])
    }


def unresolved_twins(keys):
    """Inferred twins the provider does not define although the sync entry is wrapped, mapped to that entry.

    A provider release that renames or drops the async method would otherwise
    take the twin out of ``missing_twins`` and leave the check green.
    """
    return {
        twin: key for twin, key in _twins_of_resolvable_sync_entries(keys)
        if not resolves(split_key(twin)[1])
    }


# --- Default-client bound methods ----------------------------------------------------

def class_level_targets(keys):
    """For each target wrapped as a bound method of a default client, the class attribute behind it."""
    targets = {}
    for key in keys:
        try:
            live = resolve(split_key(key)[1])
        except LookupError:
            continue
        _, inner = wrapper_chain(live)
        if inspect.ismethod(inner) and not isinstance(inner.__self__, type):
            cls = type(inner.__self__)
            targets[f"{cls.__module__}.{cls.__qualname__}.{inner.__name__}"] = (cls, inner.__name__, key)
    return targets


def uncovered_class_methods(keys):
    return {
        path: key for path, (cls, name, key) in class_level_targets(keys).items()
        if not is_ours(inspect.getattr_static(cls, name))
    }


# --- Tests ---------------------------------------------------------------------------

def _registered_param(key):
    marks = []
    if key in UNRESOLVED_ALLOWED:
        ticket, reason = UNRESOLVED_ALLOWED[key]
        marks.append(pytest.mark.xfail(strict=True, raises=LookupError, reason=f"{ticket}: {reason}"))
    return pytest.param(key, id=key, marks=marks)


@pytest.mark.parametrize("key", [_registered_param(key) for key in registered_keys()])
def test_registered_target_is_wrapped_by_its_middleware(key):
    live = resolve(split_key(key)[1])
    owner = expected_owner(key)
    assert isinstance(live, WRAPPER_TYPES), f"{key} is registered but the live attribute is {type(live).__name__}"
    assert is_ours(live, owner), f"{key} is not wrapped by {owner}; wrapper modules: {wrapper_owners(live)}"


def test_the_registry_is_populated_for_every_installed_provider():
    roots = {root_of(split_key(key)[1]) for key in registered_keys()}
    assert SDKS_IN_THIS_ENVIRONMENT <= roots


def test_every_registered_provider_has_a_twin_rule():
    roots = {root_of(split_key(key)[1]) for key in registered_keys()}
    assert None not in roots
    assert roots <= set(TWIN_RULES)


def test_every_wrapped_sync_entry_point_has_its_async_twin_registered():
    unexplained = {twin: key for twin, key in missing_twins(registered_keys()).items() if twin not in TWIN_ALLOWED}
    assert unexplained == {}, "register the async twin or allow-list it with a ticket: " + repr(unexplained)


def test_every_inferred_async_twin_exists_in_the_provider():
    unexplained = {
        twin: key for twin, key in unresolved_twins(registered_keys()).items()
        if twin not in UNRESOLVED_TWIN_ALLOWED
    }
    assert unexplained == {}, (
        "the provider no longer defines the async twin these wrapped sync entries imply; "
        "wrap its replacement or allow-list it with a ticket: "
        + ", ".join(f"{key} -> expected {twin}" for twin, key in sorted(unexplained.items()))
    )


def test_default_client_methods_are_covered_at_class_level():
    unexplained = {
        path: key for path, key in uncovered_class_methods(registered_keys()).items()
        if path not in CLASS_LEVEL_ALLOWED
    }
    assert unexplained == {}, "wrap the class method or allow-list it with a ticket: " + repr(unexplained)


class TestAllowLists:
    @pytest.mark.parametrize("allow_list",
                             [UNRESOLVED_ALLOWED, TWIN_ALLOWED, UNRESOLVED_TWIN_ALLOWED, CLASS_LEVEL_ALLOWED],
                             ids=["unresolved", "twin", "unresolved_twin", "class_level"])
    def test_every_entry_names_a_ticket_and_a_reason(self, allow_list):
        for entry, (ticket, reason) in allow_list.items():
            assert TICKET_KEY.match(ticket), entry
            assert reason, entry

    def test_unresolved_entries_are_still_registered_and_still_unresolvable(self):
        for key in filter(in_this_environment, UNRESOLVED_ALLOWED):
            assert key in patch_registry._patched, f"{key} is no longer registered; drop its entry"
            assert not resolves(split_key(key)[1]), f"{key} resolves now; drop its entry"

    def test_twin_entries_still_suppress_a_missing_twin(self):
        missing = missing_twins(registered_keys())
        stale = [twin for twin in filter(in_this_environment, TWIN_ALLOWED) if twin not in missing]
        assert stale == [], "these twins are registered now; drop their entries"

    def test_unresolved_twin_entries_still_suppress_a_missing_attribute(self):
        unresolved = unresolved_twins(registered_keys())
        assert [twin for twin in UNRESOLVED_TWIN_ALLOWED if twin not in unresolved] == []

    def test_vertex_classes_without_async_still_define_none(self):
        exempt = [key for key in registered_keys() if _has_no_async_by_design(key)]
        assert [key for key in exempt if resolves(_suffix_async_twin(key))] == [], (
            "vertexai defines an async twin now; drop the class from VERTEX_CLASSES_WITHOUT_ASYNC"
        )

    def test_class_level_entries_still_suppress_an_uncovered_method(self):
        uncovered = uncovered_class_methods(registered_keys())
        assert [path for path in filter(in_this_environment, CLASS_LEVEL_ALLOWED) if path not in uncovered] == []


class TestDetectorDoesNotTrustWrappedAttribute:
    @pytest.mark.skipif("litellm" not in SDKS_IN_THIS_ENVIRONMENT,
                        reason=f"litellm is not installed in the {selected_environment_name()} environment")
    def test_litellm_acompletion_carries_wrapped_but_is_not_ours(self):
        import litellm

        if "litellm.acompletion" not in TWIN_ALLOWED:
            assert is_ours(litellm.acompletion, "revenium_middleware.litellm")
            return
        assert hasattr(litellm.acompletion, "__wrapped__"), "litellm's own decorator sets __wrapped__"
        assert not is_ours(litellm.acompletion)
        assert "litellm.acompletion" in missing_twins(registered_keys())

    def test_a_functools_wraps_decorator_is_not_mistaken_for_our_wrap(self):
        import functools

        def original():
            return None

        @functools.wraps(original)
        def decorated():
            return original()

        assert hasattr(decorated, "__wrapped__")
        assert not is_ours(decorated)


SYNTHETIC_MODULE = "openai.resources.revenium_coverage_synthetic"


@pytest.fixture
def synthetic_provider_resource():
    return _synthetic_provider(async_defines_create=True)


@pytest.fixture
def synthetic_provider_without_async_create():
    return _synthetic_provider(async_defines_create=False)


def _synthetic_provider(async_defines_create):
    module = types.ModuleType(SYNTHETIC_MODULE)

    class Widgets:
        def create(self):
            return "sync"

    class AsyncWidgets:
        pass

    if async_defines_create:
        async def create(self):
            return "async"
        AsyncWidgets.create = create

    module.Widgets, module.AsyncWidgets = Widgets, AsyncWidgets
    return module


@pytest.fixture
def register_synthetic():
    added = []
    installed = []

    def register(module, path):
        if not installed:
            sys.modules[SYNTHETIC_MODULE] = module
            installed.append(module)
        assert patch_registry.register_patch(path)
        added.append(path)
        wrapt.wrap_function_wrapper(module, path.rpartition(f"{SYNTHETIC_MODULE}.")[2],
                                    lambda wrapped, instance, args, kwargs: wrapped(*args, **kwargs))

    yield register
    with patch_registry._lock:
        patch_registry._patched.difference_update(added)
    sys.modules.pop(SYNTHETIC_MODULE, None)


class TestNewWrapWithoutTwinFails:
    SYNC = f"{SYNTHETIC_MODULE}.Widgets.create"
    TWIN = f"{SYNTHETIC_MODULE}.AsyncWidgets.create"

    def test_a_new_sync_wrap_with_no_twin_and_no_entry_is_reported(self, synthetic_provider_resource,
                                                                   register_synthetic):
        register_synthetic(synthetic_provider_resource, self.SYNC)
        assert missing_twins(registered_keys()).get(self.TWIN) == self.SYNC
        with pytest.raises(AssertionError):
            test_every_wrapped_sync_entry_point_has_its_async_twin_registered()

    def test_registering_the_twin_clears_it(self, synthetic_provider_resource, register_synthetic):
        register_synthetic(synthetic_provider_resource, self.SYNC)
        register_synthetic(synthetic_provider_resource, self.TWIN)
        assert self.TWIN not in missing_twins(registered_keys())
        assert self.TWIN not in unresolved_twins(registered_keys())


class TestInferredTwinMissingFromProviderFails:
    SYNC = TestNewWrapWithoutTwinFails.SYNC
    TWIN = TestNewWrapWithoutTwinFails.TWIN

    def test_a_twin_the_provider_no_longer_defines_is_reported(self, synthetic_provider_without_async_create,
                                                               register_synthetic):
        register_synthetic(synthetic_provider_without_async_create, self.SYNC)
        assert self.TWIN not in missing_twins(registered_keys())
        assert unresolved_twins(registered_keys()) == {self.TWIN: self.SYNC}
        with pytest.raises(AssertionError) as failure:
            test_every_inferred_async_twin_exists_in_the_provider()
        assert f"{self.SYNC} -> expected {self.TWIN}" in str(failure.value)

    def test_an_allow_list_entry_covers_it(self, synthetic_provider_without_async_create, register_synthetic,
                                           monkeypatch):
        register_synthetic(synthetic_provider_without_async_create, self.SYNC)
        monkeypatch.setitem(UNRESOLVED_TWIN_ALLOWED, self.TWIN, ("BACK-0000", "synthetic"))
        test_every_inferred_async_twin_exists_in_the_provider()
