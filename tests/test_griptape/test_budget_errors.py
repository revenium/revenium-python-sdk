"""A spending-cap refusal reaches the caller of a Griptape structure."""
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("griptape")

from attrs import define, field  # noqa: E402
from griptape.artifacts import ErrorArtifact  # noqa: E402
from griptape.drivers.observability.base_observability_driver import BaseObservabilityDriver  # noqa: E402
from griptape.observability import Observability  # noqa: E402
from griptape.structures import Agent, Pipeline, Structure, Workflow  # noqa: E402
from griptape.tasks import PromptTask, StructureRunTask  # noqa: E402

from revenium_middleware import BudgetExceededError  # noqa: E402
from revenium_middleware.griptape import (  # noqa: E402
    ReveniumOpenAiDriver,
    disable_budget_errors,
    enable_budget_errors,
    raise_if_budget_exceeded,
)

JOIN_SECONDS = 10


@define
class RecordingObservabilityDriver(BaseObservabilityDriver):
    """Runs every observed call and records (owner class, method, exception or None)."""

    calls: list = field(factory=list, kw_only=True)

    def observe(self, call):
        entry = [type(call.instance).__name__, call.func.__name__, None]
        self.calls.append(entry)
        try:
            return call()
        except Exception as error:
            entry[2] = error
            raise

    def get_span_id(self):
        return None

    def seen(self, owner, method):
        return [entry[2] for entry in self.calls if entry[0] == owner and entry[1] == method]


@define
class FailingAfterRunObservabilityDriver(BaseObservabilityDriver):
    """A tracing driver whose first ``after_run`` raises once the call itself has run."""

    failures_left: int = field(default=1, kw_only=True)

    def observe(self, call):
        result = call()
        if call.func.__name__ == "after_run" and self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("tracing failed")
        return result

    def get_span_id(self):
        return None


@pytest.fixture(autouse=True)
def restore_griptape_observability():
    saved = Observability.get_global_driver()
    Observability.set_global_driver(None)
    yield
    Observability.set_global_driver(saved)


@pytest.fixture
def thread_errors(monkeypatch):
    """Exceptions that escaped any thread, as threading.excepthook sees them."""
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))
    return errors


@pytest.fixture
def slow_prompts(monkeypatch):
    """Each prompt attempt waits briefly, so a stream consumer can stop before the run ends."""
    base_params = ReveniumOpenAiDriver._base_params

    def slow_base_params(self, prompt_stack):
        time.sleep(0.3)
        return base_params(self, prompt_stack)

    monkeypatch.setattr(ReveniumOpenAiDriver, "_base_params", slow_base_params)


def _join_threads_started_since(before):
    for thread in set(threading.enumerate()) - before:
        thread.join(JOIN_SECONDS)
        assert not thread.is_alive(), f"thread {thread.name} did not finish"


def test_agent_run_raises_the_cap_error_after_enable_budget_errors(breached_cap, stub_provider, prompt_driver):
    enable_budget_errors()
    agent = Agent(prompt_driver=prompt_driver(stub_provider))

    with pytest.raises(BudgetExceededError) as exc_info:
        agent.run("hi")

    assert exc_info.value is agent.task.output.exception
    assert len(agent.conversation_memory.runs) == 1
    assert stub_provider.hits == 0


def test_pipeline_run_raises_the_cap_error_after_enable_budget_errors(breached_cap, stub_provider, prompt_driver):
    enable_budget_errors()
    driver = prompt_driver(stub_provider)
    pipeline = Pipeline(tasks=[PromptTask("one", prompt_driver=driver), PromptTask("two", prompt_driver=driver)])

    with pytest.raises(BudgetExceededError):
        pipeline.run()
    assert stub_provider.hits == 0


def test_workflow_run_raises_the_cap_error_after_enable_budget_errors(breached_cap, stub_provider, prompt_driver):
    enable_budget_errors()
    driver = prompt_driver(stub_provider)
    workflow = Workflow(tasks=[PromptTask("one", prompt_driver=driver, id="one"),
                               PromptTask("two", prompt_driver=driver, id="two")])

    with pytest.raises(BudgetExceededError):
        workflow.run()
    assert stub_provider.hits == 0


def test_the_cap_error_is_raised_in_a_thread_started_after_enable(breached_cap, stub_provider, prompt_driver):
    enable_budget_errors()
    outcome = []

    def run_agent():
        try:
            Agent(prompt_driver=prompt_driver(stub_provider)).run("hi")
            outcome.append(None)
        except Exception as error:
            outcome.append(error)

    worker = threading.Thread(target=run_agent)
    worker.start()
    worker.join(JOIN_SECONDS)

    assert not worker.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], BudgetExceededError)


@pytest.mark.skipif(not hasattr(Structure, "run_stream"), reason="this Griptape has no run_stream")
def test_run_stream_finishes_and_raise_if_budget_exceeded_raises_after_it(
        breached_cap, stub_provider, prompt_driver, thread_errors):
    enable_budget_errors()
    agent = Agent(prompt_driver=prompt_driver(stub_provider))
    events = []
    consumer = threading.Thread(target=lambda: events.extend(agent.run_stream("hi")), daemon=True)

    consumer.start()
    consumer.join(JOIN_SECONDS)

    assert not consumer.is_alive(), "run_stream did not finish"
    assert events
    assert thread_errors == []
    with pytest.raises(BudgetExceededError):
        raise_if_budget_exceeded(agent)


@pytest.mark.skipif(not hasattr(Structure, "run_stream"), reason="this Griptape has no run_stream")
def test_an_abandoned_run_stream_leaks_no_thread_exception(
        breached_cap, stub_provider, prompt_driver, slow_prompts, thread_errors):
    enable_budget_errors()
    agent = Agent(prompt_driver=prompt_driver(stub_provider))
    before = set(threading.enumerate())

    stream = agent.run_stream("hi")
    next(stream)
    stream.close()
    _join_threads_started_since(before)

    assert thread_errors == []
    with pytest.raises(BudgetExceededError):
        raise_if_budget_exceeded(agent)


@pytest.mark.skipif(not hasattr(Structure, "run_stream"), reason="this Griptape has no run_stream")
def test_a_plain_run_after_a_failed_run_stream_still_raises_the_cap_error(
        breached_cap, stub_provider, prompt_driver, thread_errors):
    enable_budget_errors(delegate=FailingAfterRunObservabilityDriver())
    agent = Agent(prompt_driver=prompt_driver(stub_provider))
    before = set(threading.enumerate())

    assert list(agent.run_stream("hi"))
    _join_threads_started_since(before)
    assert [type(error) for error in thread_errors] == [RuntimeError]

    with pytest.raises(BudgetExceededError):
        agent.run("hi")


def test_a_provider_error_still_returns_an_error_artifact_after_enable(failing_provider, prompt_driver):
    enable_budget_errors()
    agent = Agent(prompt_driver=prompt_driver(failing_provider))

    assert agent.run("hi") is agent
    assert isinstance(agent.output, ErrorArtifact)
    assert not isinstance(agent.output.exception, BudgetExceededError)


def test_a_successful_run_returns_the_agent_after_enable(stub_provider, prompt_driver):
    enable_budget_errors()
    agent = Agent(prompt_driver=prompt_driver(stub_provider))

    assert agent.run("hi") is agent
    assert agent.output.value == "ok"


def test_a_delegate_driver_still_observes_every_call(breached_cap, stub_provider, prompt_driver):
    recorder = RecordingObservabilityDriver()
    enable_budget_errors(delegate=recorder)
    agent = Agent(prompt_driver=prompt_driver(stub_provider))

    with pytest.raises(BudgetExceededError) as exc_info:
        agent.run("hi")

    for method in ("before_run", "try_run", "after_run"):
        assert recorder.seen("Agent", method) == [None], method
    assert recorder.seen("Agent", "run") == [exc_info.value]
    assert recorder.seen("ReveniumOpenAiDriver", "run")
    assert recorder.seen("ReveniumOpenAiDriver", "try_run")


def test_an_existing_global_driver_is_wrapped_not_replaced(breached_cap, stub_provider, prompt_driver):
    recorder = RecordingObservabilityDriver()
    Observability.set_global_driver(recorder)
    enable_budget_errors()

    with pytest.raises(BudgetExceededError):
        Agent(prompt_driver=prompt_driver(stub_provider)).run("hi")
    assert recorder.seen("Agent", "run")


def test_enabling_twice_changes_nothing():
    recorder = RecordingObservabilityDriver()
    enable_budget_errors(delegate=recorder)
    enabled = Observability.get_global_driver()

    enable_budget_errors()
    assert Observability.get_global_driver() is enabled
    enable_budget_errors(delegate=recorder)
    assert Observability.get_global_driver() is enabled


def test_a_conflicting_delegate_is_refused():
    enable_budget_errors(delegate=RecordingObservabilityDriver())
    enabled = Observability.get_global_driver()
    with pytest.raises(ValueError):
        enable_budget_errors(delegate=RecordingObservabilityDriver())
    assert Observability.get_global_driver() is enabled

    disable_budget_errors()
    customer_driver = RecordingObservabilityDriver()
    Observability.set_global_driver(customer_driver)
    with pytest.raises(ValueError, match="RecordingObservabilityDriver"):
        enable_budget_errors(delegate=RecordingObservabilityDriver())
    assert Observability.get_global_driver() is customer_driver


def test_disable_budget_errors_restores_the_previous_driver():
    customer_driver = RecordingObservabilityDriver()
    Observability.set_global_driver(customer_driver)
    enable_budget_errors()

    disable_budget_errors()
    assert Observability.get_global_driver() is customer_driver
    disable_budget_errors()
    assert Observability.get_global_driver() is customer_driver

    Observability.set_global_driver(None)
    enable_budget_errors()
    disable_budget_errors()
    assert Observability.get_global_driver() is None

    enable_budget_errors()
    replacement = RecordingObservabilityDriver()
    Observability.set_global_driver(replacement)
    disable_budget_errors()
    assert Observability.get_global_driver() is replacement


def test_importing_the_griptape_integration_installs_nothing():
    check = ("import revenium_middleware.griptape\n"
             "from griptape.observability import Observability\n"
             "assert Observability.get_global_driver() is None\n")
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]


def test_a_nested_structure_run_refusal_reaches_the_outer_caller(breached_cap, stub_provider, prompt_driver):
    from griptape.drivers.structure_run.local_structure_run_driver import LocalStructureRunDriver

    enable_budget_errors()
    driver = prompt_driver(stub_provider)
    nested = StructureRunTask(("hi",), structure_run_driver=LocalStructureRunDriver(
        create_structure=lambda: Agent(prompt_driver=driver)))

    with pytest.raises(BudgetExceededError):
        Pipeline(tasks=[nested]).run()
    assert stub_provider.hits == 0


def test_leaving_an_observability_block_turns_budget_errors_off(breached_cap, stub_provider, prompt_driver):
    with Observability(observability_driver=RecordingObservabilityDriver()):
        enable_budget_errors()
        with pytest.raises(BudgetExceededError):
            Agent(prompt_driver=prompt_driver(stub_provider)).run("hi")

    assert Observability.get_global_driver() is None


def test_an_observability_block_after_enable_is_refused():
    enable_budget_errors()

    with pytest.raises(ValueError):
        with Observability(observability_driver=RecordingObservabilityDriver()):
            pass


def test_raise_if_budget_exceeded_reraises_the_same_error_object(breached_cap, stub_provider, prompt_driver):
    agent = Agent(prompt_driver=prompt_driver(stub_provider))
    agent.run("hi")

    with pytest.raises(BudgetExceededError) as exc_info:
        raise_if_budget_exceeded(agent)
    assert exc_info.value is agent.task.output.exception


def test_raise_if_budget_exceeded_is_a_no_op_after_success(stub_provider, prompt_driver):
    agent = Agent(prompt_driver=prompt_driver(stub_provider))
    agent.run("hi")

    assert agent.output.value == "ok"
    raise_if_budget_exceeded(agent)


def test_raise_if_budget_exceeded_ignores_other_errors(failing_provider, prompt_driver):
    agent = Agent(prompt_driver=prompt_driver(failing_provider))
    agent.run("hi")

    assert isinstance(agent.output, ErrorArtifact)
    raise_if_budget_exceeded(agent)
