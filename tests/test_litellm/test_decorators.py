"""
Tests for decorator-based metadata injection.

Tests cover:
- Static agent/task_type decorators
- Dynamic extraction from arguments
- Dynamic extraction from attributes
- Sync and async function support
- Nested decorators
- Error handling
"""

import pytest
import asyncio
import contextvars
import inspect
import json
import threading
from unittest.mock import patch

from revenium_middleware.litellm.client.decorators import (
    track_agent, track_job, track_organization, track_product, track_quality, track_subscriber,
    track_subscription, track_task, track_trace,
)
from revenium_middleware.litellm.client.context import metadata_context


class TestTrackAgent:
    """Test suite for track_agent decorator."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    def test_static_agent_sync(self):
        """Test static agent name with sync function."""
        @track_agent("Test Agent")
        def my_function():
            return metadata_context.get()
        
        result = my_function()
        assert result == {"agent": "Test Agent"}
    
    @pytest.mark.asyncio
    async def test_static_agent_async(self):
        """Test static agent name with async function."""
        @track_agent("Async Agent")
        async def my_async_function():
            return metadata_context.get()
        
        result = await my_async_function()
        assert result == {"agent": "Async Agent"}
    
    def test_agent_from_arg_positional(self):
        """Test extracting agent from positional argument."""
        @track_agent(name_from_arg="agent_name")
        def my_function(agent_name, data):
            return metadata_context.get()
        
        result = my_function("Dynamic Agent", "some data")
        assert result == {"agent": "Dynamic Agent"}
    
    def test_agent_from_arg_keyword(self):
        """Test extracting agent from keyword argument."""
        @track_agent(name_from_arg="agent_name")
        def my_function(data, agent_name=None):
            return metadata_context.get()
        
        result = my_function("data", agent_name="Keyword Agent")
        assert result == {"agent": "Keyword Agent"}
    
    def test_agent_from_attr(self):
        """Test extracting agent from object attribute."""
        class MyClass:
            def __init__(self, name):
                self.name = name
            
            @track_agent(name_from_attr="name")
            def execute(self):
                return metadata_context.get()
        
        obj = MyClass("Object Agent")
        result = obj.execute()
        assert result == {"agent": "Object Agent"}
    
    def test_no_agent_source_error(self):
        """Test that error is raised when no agent source specified."""
        with pytest.raises(ValueError, match="Must specify"):
            @track_agent()
            def my_function():
                pass
    
    def test_multiple_sources_error(self):
        """Test that error is raised when multiple sources specified."""
        with pytest.raises(ValueError, match="Can only specify one"):
            @track_agent("Static", name_from_arg="arg")
            def my_function():
                pass
    
    def test_missing_argument_error(self):
        """Test that error is raised when argument not found."""
        @track_agent(name_from_arg="missing_arg")
        def my_function(other_arg):
            return metadata_context.get()
        
        with pytest.raises(ValueError, match="not found"):
            my_function("value")
    
    def test_missing_attribute_error(self):
        """Test that error is raised when attribute not found."""
        class MyClass:
            @track_agent(name_from_attr="missing_attr")
            def execute(self):
                return metadata_context.get()
        
        obj = MyClass()
        with pytest.raises(ValueError, match="does not have attribute"):
            obj.execute()
    
    def test_context_restored_after_function(self):
        """Test that context is restored after decorated function exits."""
        @track_agent("Temporary Agent")
        def my_function():
            return metadata_context.get()
        
        # Before call
        assert metadata_context.get() == {}
        
        # During call
        result = my_function()
        assert result == {"agent": "Temporary Agent"}
        
        # After call
        assert metadata_context.get() == {}


class TestTrackTask:
    """Test suite for track_task decorator."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    def test_static_task_type_sync(self):
        """Test static task type with sync function."""
        @track_task("research")
        def my_function():
            return metadata_context.get()
        
        result = my_function()
        assert result == {"task_type": "research"}
    
    @pytest.mark.asyncio
    async def test_static_task_type_async(self):
        """Test static task type with async function."""
        @track_task("async_research")
        async def my_async_function():
            return metadata_context.get()
        
        result = await my_async_function()
        assert result == {"task_type": "async_research"}
    
    def test_task_type_from_arg(self):
        """Test extracting task type from argument."""
        @track_task(type_from_arg="operation")
        def my_function(operation, data):
            return metadata_context.get()
        
        result = my_function("analysis", "some data")
        assert result == {"task_type": "analysis"}
    
    def test_task_type_from_attr(self):
        """Test extracting task type from object attribute."""
        class MyTask:
            def __init__(self, task_type):
                self.task_type = task_type
            
            @track_task(type_from_attr="task_type")
            def execute(self):
                return metadata_context.get()
        
        task = MyTask("custom_task")
        result = task.execute()
        assert result == {"task_type": "custom_task"}


class TestTrackJob:
    """Test suite for track_job decorator (BACK-777 Part 1.6)."""

    def setup_method(self):
        metadata_context.clear()

    def teardown_method(self):
        metadata_context.clear()

    def test_static_all_fields_sync(self):
        @track_job(job_id="loan-app-12345", name="Process Loan", type="loan_processing", version="1.2.0")
        def process():
            return metadata_context.get()

        result = process()
        assert result["agentic_job_id"] == "loan-app-12345"
        assert result["agentic_job_name"] == "Process Loan"
        assert result["agentic_job_type"] == "loan_processing"
        assert result["agentic_job_version"] == "1.2.0"

    def test_static_job_id_only(self):
        @track_job(job_id="job-1")
        def process():
            return metadata_context.get()

        result = process()
        assert result["agentic_job_id"] == "job-1"
        assert "agentic_job_name" not in result
        assert "agentic_job_type" not in result
        assert "agentic_job_version" not in result

    @pytest.mark.asyncio
    async def test_static_async(self):
        @track_job(job_id="job-async", type="support")
        async def process():
            return metadata_context.get()

        result = await process()
        assert result["agentic_job_id"] == "job-async"
        assert result["agentic_job_type"] == "support"

    def test_dynamic_extraction_from_args(self):
        @track_job(job_id_from_arg="job_id", type_from_arg="job_type")
        def process(job_id, job_type, data):
            return metadata_context.get()

        result = process("loan-77", "loan_processing", {"x": 1})
        assert result["agentic_job_id"] == "loan-77"
        assert result["agentic_job_type"] == "loan_processing"

    def test_dynamic_none_job_id_raises(self):
        @track_job(job_id_from_arg="job_id")
        def process(job_id, data):
            return metadata_context.get()

        with pytest.raises(ValueError, match="job_id"):
            process(None, {"x": 1})

    def test_dynamic_none_type_is_omitted(self):
        @track_job(job_id_from_arg="job_id", type_from_arg="job_type")
        def process(job_id, job_type):
            return metadata_context.get()

        result = process("job-9", None)
        assert result["agentic_job_id"] == "job-9"
        assert "agentic_job_type" not in result

    def test_dynamic_extraction_through_kwargs_catch_all(self):
        @track_job(job_id_from_arg="job_id")
        def process(**kw):
            return metadata_context.get()

        result = process(job_id="j-1")
        assert result["agentic_job_id"] == "j-1"

    def test_dynamic_none_through_kwargs_catch_all_raises(self):
        @track_job(job_id_from_arg="job_id")
        def process(**kw):
            return metadata_context.get()

        with pytest.raises(ValueError, match="job_id"):
            process(job_id=None)

    def test_context_restored_after_call(self):
        @track_job(job_id="scoped-job")
        def process():
            return metadata_context.get()

        process()
        assert "agentic_job_id" not in metadata_context.get()

    def test_requires_exactly_one_job_id_source(self):
        with pytest.raises(ValueError):
            track_job()
        with pytest.raises(ValueError):
            track_job(job_id="a", job_id_from_arg="b")

    def test_type_sources_mutually_exclusive(self):
        with pytest.raises(ValueError):
            track_job(job_id="a", type="t", type_from_arg="job_type")

    def test_metadata_context_passthrough_for_job_fields(self):
        # Ticket AC 1.7: MetadataContext is field-agnostic — job fields flow with no
        # changes to the context manager itself.
        with metadata_context.set(agentic_job_id="ctx-77", agentic_job_type="support"):
            current = metadata_context.get()
            assert current["agentic_job_id"] == "ctx-77"
            assert current["agentic_job_type"] == "support"
        assert "agentic_job_id" not in metadata_context.get()


class TestNestedDecorators:
    """Test suite for nested decorator usage."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    def test_agent_and_task_decorators(self):
        """Test using both agent and task decorators together."""
        @track_agent("Test Agent")
        @track_task("research")
        def my_function():
            return metadata_context.get()
        
        result = my_function()
        assert result == {
            "agent": "Test Agent",
            "task_type": "research"
        }
    
    def test_decorator_order_doesnt_matter(self):
        """Test that decorator order doesn't affect result."""
        @track_task("research")
        @track_agent("Test Agent")
        def my_function():
            return metadata_context.get()
        
        result = my_function()
        assert result == {
            "agent": "Test Agent",
            "task_type": "research"
        }
    
    def test_decorators_with_existing_context(self):
        """Test decorators merge with existing context."""
        metadata_context.update(trace_id="abc-123")
        
        @track_agent("Test Agent")
        @track_task("research")
        def my_function():
            return metadata_context.get()
        
        result = my_function()
        assert result == {
            "trace_id": "abc-123",
            "agent": "Test Agent",
            "task_type": "research"
        }
        
        # After function, only trace_id remains
        assert metadata_context.get() == {"trace_id": "abc-123"}
    
    @pytest.mark.asyncio
    async def test_nested_async_decorators(self):
        """Test nested decorators with async function."""
        @track_agent("Async Agent")
        @track_task("async_task")
        async def my_async_function():
            await asyncio.sleep(0.01)
            return metadata_context.get()
        
        result = await my_async_function()
        assert result == {
            "agent": "Async Agent",
            "task_type": "async_task"
        }


class TestDecoratorWithContextManager:
    """Test decorators used together with context managers."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    def test_decorator_inside_context_manager(self):
        """Test decorator used inside context manager."""
        @track_agent("Agent1")
        def my_function():
            return metadata_context.get()
        
        with metadata_context.set(trace_id="abc-123"):
            result = my_function()
            # Both context manager and decorator metadata present
            assert result == {
                "trace_id": "abc-123",
                "agent": "Agent1"
            }
        
        # After exiting context manager, all cleared
        assert metadata_context.get() == {}
    
    def test_context_manager_inside_decorator(self):
        """Test context manager used inside decorated function."""
        @track_agent("Agent1")
        def my_function():
            with metadata_context.set(task_type="research"):
                return metadata_context.get()
        
        result = my_function()
        # Both decorator and inner context manager metadata present
        assert result == {
            "agent": "Agent1",
            "task_type": "research"
        }


MESSAGES = [{"role": "user", "content": "hi there"}]


def _completion_kwargs(**extra):
    return {"model": "gpt-4o-mini", "messages": MESSAGES, "mock_response": "hello world", **extra}


def _run_in_own_thread(coro):
    thread = threading.Thread(target=asyncio.run, args=(coro,))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def litellm():
    return pytest.importorskip("litellm")


@pytest.fixture
def payloads(litellm):
    from revenium_middleware.litellm.client import middleware as mw

    recorded = []
    with patch.object(mw, "run_async_in_thread", side_effect=_run_in_own_thread), \
            patch.object(mw, "submit_ai_event", side_effect=lambda op, args, **kwargs: recorded.append(args)):
        yield recorded


TRACKERS = [
    pytest.param(lambda: track_agent("GenAgent"), lambda p: p["agent"], "GenAgent", id="track_agent"),
    pytest.param(lambda: track_task("gen-task"), lambda p: p["task_type"], "gen-task", id="track_task"),
    pytest.param(
        lambda: track_job(job_id="job-gen-1"), lambda p: (p["extra_body"] or {}).get("agenticJobId"), "job-gen-1",
        id="track_job",
    ),
    pytest.param(lambda: track_trace("gen-trace"), lambda p: p["trace_id"], "gen-trace", id="track_trace"),
    pytest.param(
        lambda: track_organization("GenOrg"), lambda p: p["organization_name"], "GenOrg", id="track_organization",
    ),
    pytest.param(
        lambda: track_subscription("sub-gen"), lambda p: p["subscription_id"], "sub-gen", id="track_subscription",
    ),
    pytest.param(lambda: track_product("gen-product"), lambda p: p["product_name"], "gen-product", id="track_product"),
    pytest.param(
        lambda: track_subscriber(subscriber_id="sub-1"), lambda p: (p["subscriber"] or {}).get("id"), "sub-1",
        id="track_subscriber",
    ),
    pytest.param(lambda: track_quality(0.75), lambda p: p["response_quality_score"], 0.75, id="track_quality"),
]


def _agent():
    return metadata_context.get().get("agent")


class TestTrackSubscriberWithoutValues:
    """A subscriber decorator with nothing to set leaves the body's own metadata changes alone."""

    def setup_method(self):
        metadata_context.clear()

    def teardown_method(self):
        metadata_context.clear()

    def test_sync_body_metadata_updates_survive_an_empty_track_subscriber(self):
        @track_subscriber()
        def work():
            metadata_context.update(some_key="v")

        work()

        assert metadata_context.get().get("some_key") == "v"

    def test_async_body_metadata_updates_survive_an_empty_track_subscriber(self):
        @track_subscriber()
        async def work():
            metadata_context.update(some_key="v")

        async def drive():
            await work()
            return metadata_context.get()

        assert asyncio.run(drive()).get("some_key") == "v"


class TestTrackingDecoratorsOnGenerators:
    """The tags stay on every LiteLLM call a decorated generator body makes, one step at a time."""

    def setup_method(self):
        metadata_context.clear()

    def teardown_method(self):
        metadata_context.clear()

    @pytest.mark.parametrize("make_decorator,field,expected", TRACKERS)
    def test_calls_inside_a_decorated_sync_generator_carry_the_decorators_field(
        self, litellm, payloads, make_decorator, field, expected
    ):
        @make_decorator()
        def stream():
            for index in range(2):
                litellm.completion(**_completion_kwargs())
                yield index

        assert list(stream()) == [0, 1]

        assert len(payloads) == 2
        assert [field(p) for p in payloads] == [expected, expected]

    @pytest.mark.parametrize("make_decorator,field,expected", TRACKERS)
    def test_calls_inside_a_decorated_async_generator_carry_the_decorators_field(
        self, litellm, payloads, make_decorator, field, expected
    ):
        @make_decorator()
        async def stream():
            for index in range(2):
                await litellm.acompletion(**_completion_kwargs())
                yield index

        async def drain():
            return [item async for item in stream()]

        assert asyncio.run(drain()) == [0, 1]

        assert len(payloads) == 2
        assert [field(p) for p in payloads] == [expected, expected]

    @pytest.mark.parametrize("kind", ["sync", "async"])
    def test_caller_context_has_no_decorator_field_between_items_or_after_the_stream(self, kind):
        seen, caller = [], []

        if kind == "sync":
            @track_agent("A")
            def steps(fail=False):
                for index in range(3):
                    seen.append(_agent())
                    if fail and index == 1:
                        raise RuntimeError("mid-stream")
                    yield index

            for _ in steps():
                caller.append(metadata_context.get())
            caller.append(metadata_context.get())

            with pytest.raises(RuntimeError):
                for _ in steps(fail=True):
                    caller.append(metadata_context.get())
            caller.append(metadata_context.get())

            closed = steps()
            next(closed)
            closed.close()
            caller.append(metadata_context.get())
        else:
            @track_agent("A")
            async def steps(fail=False):
                for index in range(3):
                    seen.append(_agent())
                    if fail and index == 1:
                        raise RuntimeError("mid-stream")
                    yield index

            async def drive():
                async for _ in steps():
                    caller.append(metadata_context.get())
                caller.append(metadata_context.get())

                with pytest.raises(RuntimeError):
                    async for _ in steps(fail=True):
                        caller.append(metadata_context.get())
                caller.append(metadata_context.get())

                closed = steps()
                await closed.__anext__()
                await closed.aclose()
                caller.append(metadata_context.get())

            asyncio.run(drive())

        assert seen == ["A"] * 6
        assert caller == [{}] * len(caller)
        assert metadata_context.get() == {}

    @pytest.mark.parametrize("kind", ["sync", "async"])
    def test_send_throw_and_close_reach_the_decorated_generator_body(self, kind):
        seen = []

        if kind == "sync":
            @track_agent("GenAgent")
            def conversation():
                try:
                    received = yield "first"
                    seen.append(("sent", received, _agent()))
                    try:
                        yield "second"
                    except KeyError:
                        seen.append(("thrown", _agent()))
                    yield "third"
                    yield "unreached"
                finally:
                    seen.append(("finally", _agent()))

            generator = conversation()
            assert next(generator) == "first"
            assert generator.send("hello") == "second"
            assert generator.throw(KeyError("boom")) == "third"
            generator.close()
        else:
            @track_agent("GenAgent")
            async def conversation():
                try:
                    received = yield "first"
                    seen.append(("sent", received, _agent()))
                    try:
                        yield "second"
                    except KeyError:
                        seen.append(("thrown", _agent()))
                    yield "third"
                    yield "unreached"
                finally:
                    seen.append(("finally", _agent()))

            async def drive():
                generator = conversation()
                assert await generator.__anext__() == "first"
                assert await generator.asend("hello") == "second"
                assert await generator.athrow(KeyError("boom")) == "third"
                await generator.aclose()

            asyncio.run(drive())

        assert seen == [("sent", "hello", "GenAgent"), ("thrown", "GenAgent"), ("finally", "GenAgent")]
        assert metadata_context.get() == {}

    def test_stacked_tracking_decorators_on_a_generator_carry_both_fields(self, litellm, payloads):
        @track_agent("StackAgent")
        @track_trace("stack-trace")
        def stream():
            litellm.completion(**_completion_kwargs())
            yield

        list(stream())

        assert len(payloads) == 1
        assert (payloads[0]["agent"], payloads[0]["trace_id"]) == ("StackAgent", "stack-trace")

    def test_dynamic_value_is_resolved_once_per_generator(self, litellm, payloads):
        class Agent:
            name = "first"

            @track_agent(name_from_attr="name")
            def stream(self):
                for _ in range(2):
                    litellm.completion(**_completion_kwargs())
                    yield

        agent = Agent()
        generator = agent.stream()
        next(generator)
        agent.name = "second"
        list(generator)

        assert [p["agent"] for p in payloads] == ["first", "first"]

    @pytest.mark.parametrize("make_decorator,field,expected", TRACKERS)
    def test_a_tracked_generator_behind_a_third_party_decorator_carries_the_field(
        self, litellm, payloads, make_decorator, field, expected
    ):
        """BACK-4325: a retry, tracing or caching decorator under ours keeps the stream's calls tagged."""
        def third_party(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        @make_decorator()
        @third_party
        def stream():
            for index in range(2):
                litellm.completion(**_completion_kwargs())
                yield index

        assert list(stream()) == [0, 1]

        assert len(payloads) == 2
        assert [field(p) for p in payloads] == [expected, expected]

    @pytest.mark.parametrize("make_decorator,field,expected", TRACKERS)
    def test_tracked_generators_stay_generator_functions(self, make_decorator, field, expected):
        @make_decorator()
        def stream():
            yield

        @make_decorator()
        async def async_stream():
            yield

        assert inspect.isgeneratorfunction(stream)
        assert inspect.isasyncgenfunction(async_stream)

    def test_each_step_in_a_fresh_context_keeps_the_tag(self, litellm, payloads):
        @track_agent("CopiedAgent")
        def stream():
            for index in range(2):
                litellm.completion(**_completion_kwargs())
                yield index

        generator = stream()
        for _ in range(2):
            contextvars.copy_context().run(next, generator)
            assert metadata_context.get() == {}
        contextvars.copy_context().run(generator.close)

        assert [p["agent"] for p in payloads] == ["CopiedAgent", "CopiedAgent"]

    def test_a_stream_yielded_by_the_body_and_drained_by_the_caller_keeps_the_tag(self, litellm, payloads):
        @track_agent("StreamAgent")
        def stream():
            yield litellm.completion(**_completion_kwargs(stream=True))

        for litellm_stream in stream():
            list(litellm_stream)

        assert len(payloads) == 1
        assert payloads[0]["agent"] == "StreamAgent"
        assert payloads[0]["is_streamed"] is True

    def test_the_return_value_of_a_decorated_generator_reaches_a_yield_from_caller(self):
        @track_agent("ReturnAgent")
        def inner():
            yield _agent()
            return "done"

        def outer():
            result = yield from inner()
            yield result

        assert list(outer()) == ["ReturnAgent", "done"]


class TestEmptyPerCallMetadata:
    """BACK-4322: an empty per-call field leaves a tracking decorator's value on the record."""

    def setup_method(self):
        metadata_context.clear()

    def teardown_method(self):
        metadata_context.clear()

    @pytest.mark.parametrize("blank", [None, ""], ids=["none", "empty"])
    def test_an_empty_usage_metadata_field_keeps_the_tracking_decorators_value(self, litellm, payloads, blank):
        @track_organization("acme")
        def handle():
            litellm.completion(**_completion_kwargs(usage_metadata={"trace_id": "t-1", "organization_name": blank}))

        handle()

        assert len(payloads) == 1
        assert payloads[0]["organization_name"] == "acme"
        assert payloads[0]["trace_id"] == "t-1"

    def test_a_real_usage_metadata_value_still_overrides_the_tracking_decorator(self, litellm, payloads):
        @track_organization("acme")
        def handle():
            litellm.completion(**_completion_kwargs(usage_metadata={"organization_name": "per-call"}))

        handle()

        assert len(payloads) == 1
        assert payloads[0]["organization_name"] == "per-call"


ARGUMENT_TRACKERS = [
    pytest.param(lambda: track_agent(name_from_arg="value"), "agent", id="track_agent"),
    pytest.param(lambda: track_task(type_from_arg="value"), "task_type", id="track_task"),
    pytest.param(lambda: track_trace(id_from_arg="value"), "trace_id", id="track_trace"),
    pytest.param(lambda: track_organization(name_from_arg="value"), "organization_name", id="track_organization"),
    pytest.param(lambda: track_subscription(id_from_arg="value"), "subscription_id", id="track_subscription"),
    pytest.param(lambda: track_product(name_from_arg="value"), "product_name", id="track_product"),
]
PASSING = pytest.mark.parametrize("passing", ["keyword", "positional"])
EMPTY = pytest.mark.parametrize("empty", [None, ""], ids=["none", "empty"])


def _call(function, value, passing):
    return function("q", value=value) if passing == "keyword" else function("q", value)


def _sends_no_field(payloads, field):
    assert len(payloads) == 1
    payload = payloads[0]
    assert field not in payload or payload[field] is None
    assert '"None"' not in json.dumps(payload, default=str)


class TestTrackingDecoratorsWithEmptyArguments:
    """BACK-4317: an argument passed as None or "" sets no tag, never the text "None"."""

    def setup_method(self):
        metadata_context.clear()

    def teardown_method(self):
        metadata_context.clear()

    @PASSING
    @EMPTY
    @pytest.mark.parametrize("make_decorator,field", ARGUMENT_TRACKERS)
    def test_an_empty_argument_sets_no_tag(self, litellm, payloads, make_decorator, field, empty, passing):
        @make_decorator()
        def answer(question, value):
            return litellm.completion(**_completion_kwargs())

        _call(answer, empty, passing)

        _sends_no_field(payloads, field)

    @PASSING
    def test_subscriber_arguments_passed_as_none_send_no_subscriber(self, litellm, payloads, passing):
        @track_subscriber(id_from_arg="user_id", email_from_arg="user_email", credential_from_arg="credential")
        def answer(question, user_id, user_email, credential):
            return litellm.completion(**_completion_kwargs())

        if passing == "keyword":
            answer("q", user_id=None, user_email=None, credential=None)
        else:
            answer("q", None, None, None)

        assert len(payloads) == 1
        assert not payloads[0].get("subscriber")
        assert '"None"' not in json.dumps(payloads[0], default=str)

    def test_an_empty_argument_keeps_the_outer_decorators_value(self, litellm, payloads):
        @track_organization("acme")
        @track_organization(name_from_arg="org")
        def answer(question, org):
            return litellm.completion(**_completion_kwargs())

        answer("q", org=None)

        assert len(payloads) == 1
        assert payloads[0]["organization_name"] == "acme"

    @EMPTY
    def test_an_attribute_set_to_an_empty_value_sets_no_tag(self, litellm, payloads, empty):
        class Agent:
            name = empty

            @track_agent(name_from_attr="name")
            def answer(self):
                return litellm.completion(**_completion_kwargs())

        Agent().answer()

        _sends_no_field(payloads, "agent")

    @PASSING
    @pytest.mark.parametrize("make_decorator,field", ARGUMENT_TRACKERS)
    def test_a_real_argument_is_still_recorded(self, litellm, payloads, make_decorator, field, passing):
        @make_decorator()
        def answer(question, value):
            return litellm.completion(**_completion_kwargs())

        _call(answer, "acme", passing)

        assert len(payloads) == 1
        assert payloads[0][field] == "acme"


class TestTrackingDecoratorsWithDefaultArguments:
    """BACK-4319: how the tagged argument is passed never makes the customer's function fail."""

    def setup_method(self):
        metadata_context.clear()

    def teardown_method(self):
        metadata_context.clear()

    @pytest.mark.parametrize("make_decorator,field", ARGUMENT_TRACKERS)
    def test_a_defaulted_argument_is_recorded(self, litellm, payloads, make_decorator, field):
        @make_decorator()
        def run(question, value="support-agent"):
            litellm.completion(**_completion_kwargs())
            return "answer"

        assert run("q") == "answer"

        assert len(payloads) == 1
        assert payloads[0][field] == "support-agent"

    @pytest.mark.parametrize("make_decorator,field", ARGUMENT_TRACKERS)
    def test_a_defaulted_none_argument_sets_no_tag(self, litellm, payloads, make_decorator, field):
        @make_decorator()
        def run(question, value=None):
            litellm.completion(**_completion_kwargs())
            return "answer"

        assert run("q") == "answer"

        _sends_no_field(payloads, field)

    @pytest.mark.parametrize("call", [lambda graded: graded("q"), lambda graded: graded("q", score=None)],
                             ids=["no_score", "score_none"])
    def test_a_quality_scored_function_without_a_score_returns_and_sends_no_score(self, litellm, payloads, call):
        @track_quality(score_from_arg="score")
        def graded(question, score=None):
            litellm.completion(**_completion_kwargs())
            return "answer"

        assert call(graded) == "answer"

        _sends_no_field(payloads, "response_quality_score")

    def test_a_non_numeric_quality_score_returns_and_sends_no_score(self, litellm, payloads, caplog):
        @track_quality(score_from_arg="score")
        def graded(question, score=None):
            litellm.completion(**_completion_kwargs())
            return "answer"

        with caplog.at_level("WARNING", logger="revenium_middleware"):
            assert graded("q", score="high") == "answer"

        _sends_no_field(payloads, "response_quality_score")
        warnings = [record.getMessage() for record in caplog.records if "quality_score" in record.getMessage()]
        assert len(warnings) == 1
        assert "high" not in warnings[0]

    def test_a_quality_score_of_zero_is_recorded_as_zero(self, litellm, payloads):
        @track_quality(score_from_arg="score")
        def graded(question, score=None):
            return litellm.completion(**_completion_kwargs())

        graded("q", score=0)

        assert len(payloads) == 1
        assert payloads[0]["response_quality_score"] == 0.0

    def test_an_argument_through_kwargs_catch_all_that_is_not_passed_sets_no_tag(self, litellm, payloads):
        @track_agent(name_from_arg="agent_name")
        def run(question, **extra):
            litellm.completion(**_completion_kwargs())
            return "answer"

        assert run("q") == "answer"

        _sends_no_field(payloads, "agent")

    def test_an_argument_through_kwargs_catch_all_that_is_passed_is_recorded(self, litellm, payloads):
        @track_agent(name_from_arg="agent_name")
        def run(question, **extra):
            return litellm.completion(**_completion_kwargs())

        run("q", agent_name="catch-all-agent")

        assert len(payloads) == 1
        assert payloads[0]["agent"] == "catch-all-agent"

    def test_a_call_missing_a_required_argument_raises_the_functions_own_error(self):
        def run(question, value):
            return "answer"

        decorated = track_agent(name_from_arg="value")(run)

        with pytest.raises(TypeError) as undecorated_error:
            run()
        with pytest.raises(TypeError) as decorated_error:
            decorated()
        assert str(decorated_error.value) == str(undecorated_error.value)

    def test_a_misnamed_argument_still_raises(self):
        @track_agent(name_from_arg="agent_nmae")
        def run(question, agent_name="a"):
            return "answer"

        with pytest.raises(ValueError, match="not found"):
            run("q")

    @pytest.mark.parametrize("decorator", [
        lambda: track_job(job_id_from_arg="job_id"),
        lambda: track_job(job_id="j-1", type_from_arg="job_type"),
    ], ids=["job_id", "job_type"])
    def test_track_job_argument_not_passed_through_kwargs_catch_all_still_raises(self, decorator):
        @decorator()
        def process(**kw):
            return metadata_context.get()

        with pytest.raises(ValueError, match="not found"):
            process()
