from revenium_middleware._core.patch_registry import is_patched, register_patch, unregister_patch

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
