import sys
import types

import pytest

from revenium_middleware._core.patch_registry import is_patched, register_patch, unregister_patch, wrap_once

KEY = "revenium_test.synthetic.Resource.create"


def test_an_unregistered_patch_is_no_longer_reported_and_can_be_registered_again():
    assert register_patch(KEY)
    unregister_patch(KEY)

    assert not is_patched(KEY)
    assert register_patch(KEY)
    unregister_patch(KEY)


def test_unregistering_an_unknown_patch_is_a_no_op():
    unregister_patch(KEY)
    assert not is_patched(KEY)


def test_wrap_once_unregisters_when_the_wrap_fails():
    """BACK-4235: a wrap that raises leaves the method unrecorded, and the error reaches the caller."""
    with pytest.raises(ImportError):
        wrap_once(KEY, "revenium_test_not_installed", "Resource.create", lambda *args: None)

    assert not is_patched(KEY)


def test_wrap_once_wraps_and_records_once(monkeypatch):
    module = types.ModuleType("revenium_test_synthetic")
    module.Resource = type("Resource", (), {"create": lambda self: "created"})
    monkeypatch.setitem(sys.modules, module.__name__, module)

    try:
        assert wrap_once(KEY, module.__name__, "Resource.create", lambda wrapped, instance, args, kwargs: "wrapped")
        assert not wrap_once(KEY, module.__name__, "Resource.create", lambda wrapped, instance, args, kwargs: "twice")

        assert is_patched(KEY)
        assert module.Resource().create() == "wrapped"
    finally:
        unregister_patch(KEY)
