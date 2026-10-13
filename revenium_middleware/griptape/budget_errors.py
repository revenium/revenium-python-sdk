"""
Spending-cap errors for Griptape structures.

Griptape turns any exception inside a task into an ``ErrorArtifact``, so an
``Agent``, ``Pipeline`` or ``Workflow`` stopped by a Revenium spending cap
returns normally with the refusal as its output. ``enable_budget_errors()``
makes ``run()`` raise that ``BudgetExceededError`` instead, through Griptape's
public observability hook; ``raise_if_budget_exceeded(structure)`` does the same
on demand. Nothing is installed on import.
"""
import functools
import threading
from typing import Optional

import attrs
from griptape.artifacts import ErrorArtifact
from griptape.drivers.observability.base_observability_driver import BaseObservabilityDriver
from griptape.observability import Observability
from griptape.structures import Structure

from revenium_middleware._core.exceptions import BudgetExceededError

_enable_lock = threading.Lock()


def _find_budget_error(structure) -> Optional[BudgetExceededError]:
    for task in structure.tasks:
        output = task.output
        if isinstance(output, ErrorArtifact) and isinstance(output.exception, BudgetExceededError):
            return output.exception
    return None


def raise_if_budget_exceeded(structure) -> None:
    """Raise the ``BudgetExceededError`` that stopped a task of ``structure``'s last run.

    Use it after ``run_stream``, or after ``run`` when ``enable_budget_errors()``
    is not in use. Re-raises the original error object; does nothing when the
    run succeeded or failed for any other reason.
    """
    error = _find_budget_error(structure)
    if error is not None:
        raise error


@attrs.define
class _BudgetErrorObservabilityDriver(BaseObservabilityDriver):
    """Raises a structure's spending-cap refusal from ``run`` and passes every call to ``delegate``."""

    delegate: Optional[BaseObservabilityDriver] = attrs.field(default=None, kw_only=True)
    _streaming: set = attrs.field(factory=set, init=False)

    def observe(self, call):
        if not isinstance(call.instance, Structure):
            return self._pass_through(call)
        name = call.func.__name__
        if name == "run_stream":
            return self._stream(call)
        if name == "run":
            return self._pass_through(self._raising_after(call))
        return self._pass_through(call)

    def get_span_id(self):
        return None if self.delegate is None else self.delegate.get_span_id()

    def _pass_through(self, call):
        return call() if self.delegate is None else self.delegate.observe(call)

    def _raising_after(self, call):
        """The same call, raising the refusal before it returns, so the delegate's span records the error."""
        structure = call.instance
        run = call.func

        @functools.wraps(run)
        def run_then_raise(*args, **kwargs):
            key = id(structure)
            try:
                result = run(*args, **kwargs)
            finally:
                streamed = key in self._streaming
                self._streaming.discard(key)
            if not streamed:
                raise_if_budget_exceeded(structure)
            return result

        return attrs.evolve(call, func=run_then_raise)

    def _stream(self, call):
        """``run_stream`` runs ``run`` in its own thread, where a raise reaches no caller; mark it to skip."""
        key = id(call.instance)
        events = self._pass_through(call)

        def marked():
            self._streaming.add(key)
            started = False
            try:
                for event in events:
                    started = True
                    yield event
            finally:
                if not started:
                    self._streaming.discard(key)
                if hasattr(events, "close"):
                    events.close()

        return marked()


def enable_budget_errors(delegate: Optional[BaseObservabilityDriver] = None) -> None:
    """Make ``run()`` on a Griptape ``Agent``, ``Pipeline`` or ``Workflow`` raise ``BudgetExceededError``.

    Call it once at startup. It sets Griptape's global observability driver,
    wrapping any driver already set (or ``delegate``) so that driver still sees
    every call. Calling it again is a no-op. ``run_stream`` never raises; call
    ``raise_if_budget_exceeded(structure)`` after the stream ends.

    Leaving a ``with Observability(...)`` or ``with GriptapeCloudStructure(observe=True)``
    block clears Griptape's global driver, which turns this off. If you use one,
    call ``enable_budget_errors()`` inside each block, after it opens; calling it
    before the block makes Griptape refuse the block. Pass ``delegate`` only for a
    driver you have already entered, or one with no enter or exit work.

    Raises:
        ValueError: ``delegate`` differs from the driver already set, or from
            the one an earlier call wrapped. Call ``disable_budget_errors()`` first.
    """
    with _enable_lock:
        current = Observability.get_global_driver()
        if isinstance(current, _BudgetErrorObservabilityDriver):
            if delegate is None or delegate is current.delegate:
                return
            raise ValueError(
                "Spending-cap errors are already enabled with a different observability driver. "
                "Call disable_budget_errors() first."
            )
        if current is not None and delegate is not None and delegate is not current:
            raise ValueError(
                f"Griptape already has a global observability driver ({type(current).__name__}). "
                "Call enable_budget_errors() with no delegate to wrap it, or pass that same driver."
            )
        Observability.set_global_driver(_BudgetErrorObservabilityDriver(
            delegate=current if delegate is None else delegate))


def disable_budget_errors() -> None:
    """Turn ``enable_budget_errors()`` off and restore the observability driver it wrapped."""
    with _enable_lock:
        current = Observability.get_global_driver()
        if isinstance(current, _BudgetErrorObservabilityDriver):
            Observability.set_global_driver(current.delegate)
