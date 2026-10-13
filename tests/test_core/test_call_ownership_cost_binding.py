"""BACK-4282: a cost-reading wrap binds the run the model call is made under.

Each test runs in a copied context, so its publications never reach another test.
"""
import contextvars

from revenium_middleware._core import call_ownership
from revenium_middleware._core.call_ownership import PERPLEXITY, bind_model_call_for_cost, publish_callback_metadata


def _in_a_fresh_context(test):
    return contextvars.Context().run(test)


def _two_open_runs():
    first = publish_callback_metadata({"agent": "first"}, frozenset({PERPLEXITY}), "run-a")
    second = publish_callback_metadata({"agent": "second"}, frozenset({PERPLEXITY}), "run-b")
    return first, second


def test_the_run_being_generated_is_bound_even_when_an_earlier_run_is_open():
    def bind():
        _, second = _two_open_runs()
        with call_ownership.generating_for_run("run-b"):
            return second, bind_model_call_for_cost(PERPLEXITY)

    second, bound = _in_a_fresh_context(bind)
    assert bound == (second,)
    assert second.cost_bound


def test_a_run_with_no_open_publication_binds_nothing():
    def bind():
        first, second = _two_open_runs()
        with call_ownership.generating_for_run("run-unknown"):
            return first, second, bind_model_call_for_cost(PERPLEXITY)

    first, second, bound = _in_a_fresh_context(bind)
    assert bound == ()
    assert not first.cost_bound and not second.cost_bound


def test_outside_a_generating_run_the_earliest_unbound_run_is_bound():
    def bind():
        first, _ = _two_open_runs()
        return first, bind_model_call_for_cost(PERPLEXITY)

    first, bound = _in_a_fresh_context(bind)
    assert bound == (first,)


def test_the_generating_run_is_forgotten_when_its_call_returns():
    def bind():
        first, _ = _two_open_runs()
        with call_ownership.generating_for_run("run-b"):
            pass
        return first, bind_model_call_for_cost(PERPLEXITY)

    first, bound = _in_a_fresh_context(bind)
    assert bound == (first,)
