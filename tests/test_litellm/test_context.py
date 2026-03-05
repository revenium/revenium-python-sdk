"""
Tests for context-based metadata injection.

Tests cover:
- Basic get/set/update/clear operations
- Context manager scoping
- Nested contexts
- Thread safety
- Async compatibility
- Metadata merging
"""

import pytest
import asyncio
import threading
from revenium_middleware.litellm.client.context import metadata_context, MetadataContext


class TestMetadataContext:
    """Test suite for MetadataContext class."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    def test_get_empty_context(self):
        """Test getting metadata from empty context."""
        result = metadata_context.get()
        assert result == {}
        assert isinstance(result, dict)
    
    def test_update_single_field(self):
        """Test updating a single metadata field."""
        metadata_context.update(agent="Test Agent")
        result = metadata_context.get()
        assert result == {"agent": "Test Agent"}
    
    def test_update_multiple_fields(self):
        """Test updating multiple metadata fields."""
        metadata_context.update(agent="Agent1", task_type="research")
        result = metadata_context.get()
        assert result == {"agent": "Agent1", "task_type": "research"}
    
    def test_update_incremental(self):
        """Test incremental updates merge with existing metadata."""
        metadata_context.update(agent="Agent1")
        metadata_context.update(task_type="research")
        metadata_context.update(trace_id="abc-123")
        
        result = metadata_context.get()
        assert result == {
            "agent": "Agent1",
            "task_type": "research",
            "trace_id": "abc-123"
        }
    
    def test_update_overwrites_existing(self):
        """Test that update overwrites existing values."""
        metadata_context.update(agent="Agent1")
        metadata_context.update(agent="Agent2")
        
        result = metadata_context.get()
        assert result == {"agent": "Agent2"}
    
    def test_clear_all(self):
        """Test clearing all metadata."""
        metadata_context.update(agent="Agent1", task_type="research")
        metadata_context.clear()
        
        result = metadata_context.get()
        assert result == {}
    
    def test_clear_specific_keys(self):
        """Test clearing specific metadata keys."""
        metadata_context.update(
            agent="Agent1",
            task_type="research",
            trace_id="abc-123"
        )
        metadata_context.clear("agent", "trace_id")
        
        result = metadata_context.get()
        assert result == {"task_type": "research"}
    
    def test_clear_nonexistent_key(self):
        """Test clearing a key that doesn't exist (should not error)."""
        metadata_context.update(agent="Agent1")
        metadata_context.clear("nonexistent")
        
        result = metadata_context.get()
        assert result == {"agent": "Agent1"}
    
    def test_get_returns_copy(self):
        """Test that get() returns a copy, not the original dict."""
        metadata_context.update(agent="Agent1")
        result1 = metadata_context.get()
        result1["modified"] = "value"
        
        result2 = metadata_context.get()
        assert "modified" not in result2
        assert result2 == {"agent": "Agent1"}


class TestContextManager:
    """Test suite for context manager functionality."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with metadata_context.set(agent="Agent1"):
            result = metadata_context.get()
            assert result == {"agent": "Agent1"}
        
        # Context should be cleared after exiting
        result = metadata_context.get()
        assert result == {}
    
    def test_context_manager_multiple_fields(self):
        """Test context manager with multiple fields."""
        with metadata_context.set(agent="Agent1", task_type="research"):
            result = metadata_context.get()
            assert result == {"agent": "Agent1", "task_type": "research"}
    
    def test_context_manager_nested(self):
        """Test nested context managers."""
        with metadata_context.set(agent="Agent1"):
            assert metadata_context.get() == {"agent": "Agent1"}
            
            with metadata_context.set(task_type="research"):
                # Inner context should merge with outer
                result = metadata_context.get()
                assert result == {"agent": "Agent1", "task_type": "research"}
            
            # After exiting inner context, should restore outer
            result = metadata_context.get()
            assert result == {"agent": "Agent1"}
        
        # After exiting all contexts, should be empty
        result = metadata_context.get()
        assert result == {}
    
    def test_context_manager_override(self):
        """Test that inner context can override outer context values."""
        with metadata_context.set(agent="Agent1", task_type="research"):
            with metadata_context.set(agent="Agent2"):
                result = metadata_context.get()
                assert result == {"agent": "Agent2", "task_type": "research"}
            
            # After exiting inner, outer value restored
            result = metadata_context.get()
            assert result == {"agent": "Agent1", "task_type": "research"}
    
    def test_context_manager_with_exception(self):
        """Test that context is restored even if exception occurs."""
        metadata_context.update(initial="value")
        
        try:
            with metadata_context.set(agent="Agent1"):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Context should be restored to pre-context-manager state
        result = metadata_context.get()
        assert result == {"initial": "value"}
    
    def test_context_manager_preserves_existing(self):
        """Test that context manager preserves existing metadata."""
        metadata_context.update(existing="value")
        
        with metadata_context.set(agent="Agent1"):
            result = metadata_context.get()
            assert result == {"existing": "value", "agent": "Agent1"}
        
        result = metadata_context.get()
        assert result == {"existing": "value"}


class TestThreadSafety:
    """Test suite for thread safety."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    def test_thread_isolation(self):
        """Test that contexts are isolated between threads."""
        results = {}
        
        def thread_func(thread_id, agent_name):
            metadata_context.update(agent=agent_name)
            # Small delay to ensure threads overlap
            import time
            time.sleep(0.01)
            results[thread_id] = metadata_context.get()
        
        threads = [
            threading.Thread(target=thread_func, args=(1, "Agent1")),
            threading.Thread(target=thread_func, args=(2, "Agent2")),
            threading.Thread(target=thread_func, args=(3, "Agent3")),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Each thread should have its own isolated context
        assert results[1] == {"agent": "Agent1"}
        assert results[2] == {"agent": "Agent2"}
        assert results[3] == {"agent": "Agent3"}


class TestAsyncCompatibility:
    """Test suite for async compatibility."""
    
    def setup_method(self):
        """Clear context before each test."""
        metadata_context.clear()
    
    def teardown_method(self):
        """Clear context after each test."""
        metadata_context.clear()
    
    @pytest.mark.asyncio
    async def test_async_context_isolation(self):
        """Test that contexts are isolated in async tasks."""
        results = {}
        
        async def async_task(task_id, agent_name):
            metadata_context.update(agent=agent_name)
            await asyncio.sleep(0.01)  # Simulate async work
            results[task_id] = metadata_context.get()
        
        # Run multiple async tasks concurrently
        await asyncio.gather(
            async_task(1, "Agent1"),
            async_task(2, "Agent2"),
            async_task(3, "Agent3"),
        )
        
        # Each task should have its own isolated context
        assert results[1] == {"agent": "Agent1"}
        assert results[2] == {"agent": "Agent2"}
        assert results[3] == {"agent": "Agent3"}
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test context manager in async context."""
        with metadata_context.set(agent="AsyncAgent"):
            await asyncio.sleep(0.01)
            result = metadata_context.get()
            assert result == {"agent": "AsyncAgent"}
        
        result = metadata_context.get()
        assert result == {}

