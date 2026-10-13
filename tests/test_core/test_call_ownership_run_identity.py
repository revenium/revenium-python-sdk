"""BACK-4309: a transport call made while langchain-core generates a run takes that run's attribution.

Each test runs in a copied context, so its publications never reach another test.
"""
import contextvars

from revenium_middleware._core import call_ownership
from revenium_middleware._core.call_ownership import OPENAI, publish_callback_metadata, take_callback_metadata


def _in_a_fresh_context(test):
    return contextvars.Context().run(test)


def _three_open_runs():
    for run_id, organization in (("ra", "org-a"), ("rb", "org-b"), ("rc", "org-c")):
        publish_callback_metadata({"organizationName": organization}, frozenset({OPENAI}), run_id)


def test_the_run_being_generated_takes_its_own_metadata_when_later_runs_are_open():
    def take():
        _three_open_runs()
        with call_ownership.generating_for_run("ra"):
            return take_callback_metadata(OPENAI)

    assert _in_a_fresh_context(take) == {"organizationName": "org-a"}


def test_a_run_with_no_open_publication_of_its_own_takes_the_latest_open_one():
    def take():
        _three_open_runs()
        with call_ownership.generating_for_run("run-of-an-inner-model"):
            return take_callback_metadata(OPENAI)

    assert _in_a_fresh_context(take) == {"organizationName": "org-c"}


def test_outside_a_generating_run_the_latest_open_run_is_taken():
    def take():
        _three_open_runs()
        return take_callback_metadata(OPENAI)

    assert _in_a_fresh_context(take) == {"organizationName": "org-c"}
