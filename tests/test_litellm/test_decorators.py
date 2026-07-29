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
from revenium_middleware.litellm.client.decorators import track_agent, track_task, track_job
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

