"""
Tests for hook system.

Tests cover:
- Hook registration and unregistration
- Hook execution order (priority)
- Hook error handling
- Hook modification of metadata
- Multiple hooks
"""

import pytest
from revenium_middleware.litellm.client.hooks import (
    register_metadata_hook,
    unregister_metadata_hook,
    clear_metadata_hooks,
    execute_metadata_hooks,
    get_registered_hooks
)


class TestHookRegistration:
    """Test suite for hook registration."""
    
    def setup_method(self):
        """Clear hooks before each test."""
        clear_metadata_hooks()
    
    def teardown_method(self):
        """Clear hooks after each test."""
        clear_metadata_hooks()
    
    def test_register_single_hook(self):
        """Test registering a single hook."""
        def my_hook(metadata):
            return metadata
        
        register_metadata_hook(my_hook)
        hooks = get_registered_hooks()
        
        assert len(hooks) == 1
        assert hooks[0] == my_hook
    
    def test_register_multiple_hooks(self):
        """Test registering multiple hooks."""
        def hook1(metadata):
            return metadata
        
        def hook2(metadata):
            return metadata
        
        def hook3(metadata):
            return metadata
        
        register_metadata_hook(hook1)
        register_metadata_hook(hook2)
        register_metadata_hook(hook3)
        
        hooks = get_registered_hooks()
        assert len(hooks) == 3
    
    def test_unregister_hook(self):
        """Test unregistering a hook."""
        def my_hook(metadata):
            return metadata
        
        register_metadata_hook(my_hook)
        assert len(get_registered_hooks()) == 1
        
        result = unregister_metadata_hook(my_hook)
        assert result is True
        assert len(get_registered_hooks()) == 0
    
    def test_unregister_nonexistent_hook(self):
        """Test unregistering a hook that wasn't registered."""
        def my_hook(metadata):
            return metadata
        
        result = unregister_metadata_hook(my_hook)
        assert result is False
    
    def test_clear_all_hooks(self):
        """Test clearing all hooks."""
        def hook1(metadata):
            return metadata
        
        def hook2(metadata):
            return metadata
        
        register_metadata_hook(hook1)
        register_metadata_hook(hook2)
        assert len(get_registered_hooks()) == 2
        
        clear_metadata_hooks()
        assert len(get_registered_hooks()) == 0


class TestHookPriority:
    """Test suite for hook priority and execution order."""
    
    def setup_method(self):
        """Clear hooks before each test."""
        clear_metadata_hooks()
    
    def teardown_method(self):
        """Clear hooks after each test."""
        clear_metadata_hooks()
    
    def test_default_priority(self):
        """Test hooks with default priority execute in registration order."""
        execution_order = []
        
        def hook1(metadata):
            execution_order.append(1)
            return metadata
        
        def hook2(metadata):
            execution_order.append(2)
            return metadata
        
        def hook3(metadata):
            execution_order.append(3)
            return metadata
        
        register_metadata_hook(hook1)
        register_metadata_hook(hook2)
        register_metadata_hook(hook3)
        
        execute_metadata_hooks({})
        assert execution_order == [1, 2, 3]
    
    def test_priority_order(self):
        """Test hooks execute in priority order (highest first)."""
        execution_order = []
        
        def low_priority(metadata):
            execution_order.append("low")
            return metadata
        
        def medium_priority(metadata):
            execution_order.append("medium")
            return metadata
        
        def high_priority(metadata):
            execution_order.append("high")
            return metadata
        
        register_metadata_hook(low_priority, priority=1)
        register_metadata_hook(high_priority, priority=100)
        register_metadata_hook(medium_priority, priority=50)
        
        execute_metadata_hooks({})
        assert execution_order == ["high", "medium", "low"]
    
    def test_same_priority_registration_order(self):
        """Test hooks with same priority execute in registration order."""
        execution_order = []
        
        def hook1(metadata):
            execution_order.append(1)
            return metadata
        
        def hook2(metadata):
            execution_order.append(2)
            return metadata
        
        register_metadata_hook(hook1, priority=10)
        register_metadata_hook(hook2, priority=10)
        
        execute_metadata_hooks({})
        assert execution_order == [1, 2]


class TestHookExecution:
    """Test suite for hook execution and metadata modification."""
    
    def setup_method(self):
        """Clear hooks before each test."""
        clear_metadata_hooks()
    
    def teardown_method(self):
        """Clear hooks after each test."""
        clear_metadata_hooks()
    
    def test_hook_adds_field(self):
        """Test hook can add new fields to metadata."""
        def add_version(metadata):
            metadata['version'] = '1.0.0'
            return metadata
        
        register_metadata_hook(add_version)
        
        result = execute_metadata_hooks({'agent': 'Test'})
        assert result == {'agent': 'Test', 'version': '1.0.0'}
    
    def test_hook_modifies_field(self):
        """Test hook can modify existing fields."""
        def uppercase_agent(metadata):
            if 'agent' in metadata:
                metadata['agent'] = metadata['agent'].upper()
            return metadata
        
        register_metadata_hook(uppercase_agent)
        
        result = execute_metadata_hooks({'agent': 'test agent'})
        assert result == {'agent': 'TEST AGENT'}
    
    def test_hook_removes_field(self):
        """Test hook can remove fields."""
        def remove_sensitive(metadata):
            metadata.pop('password', None)
            return metadata
        
        register_metadata_hook(remove_sensitive)
        
        result = execute_metadata_hooks({
            'agent': 'Test',
            'password': 'secret'
        })
        assert result == {'agent': 'Test'}
    
    def test_multiple_hooks_chain(self):
        """Test multiple hooks chain modifications."""
        def add_prefix(metadata):
            if 'agent' in metadata:
                metadata['agent'] = f"prefix_{metadata['agent']}"
            return metadata
        
        def add_suffix(metadata):
            if 'agent' in metadata:
                metadata['agent'] = f"{metadata['agent']}_suffix"
            return metadata
        
        register_metadata_hook(add_prefix, priority=2)
        register_metadata_hook(add_suffix, priority=1)
        
        result = execute_metadata_hooks({'agent': 'test'})
        assert result == {'agent': 'prefix_test_suffix'}
    
    def test_empty_metadata(self):
        """Test hooks work with empty metadata."""
        def add_default(metadata):
            metadata['default'] = 'value'
            return metadata
        
        register_metadata_hook(add_default)
        
        result = execute_metadata_hooks({})
        assert result == {'default': 'value'}
    
    def test_no_hooks_returns_original(self):
        """Test that metadata is returned unchanged when no hooks registered."""
        original = {'agent': 'Test', 'task_type': 'research'}
        result = execute_metadata_hooks(original)
        
        assert result == original
        # Should be a copy, not the same object
        assert result is not original


class TestHookErrorHandling:
    """Test suite for hook error handling."""
    
    def setup_method(self):
        """Clear hooks before each test."""
        clear_metadata_hooks()
    
    def teardown_method(self):
        """Clear hooks after each test."""
        clear_metadata_hooks()
    
    def test_hook_exception_continues_execution(self):
        """Test that exception in one hook doesn't stop others."""
        execution_order = []
        
        def good_hook1(metadata):
            execution_order.append(1)
            metadata['hook1'] = 'executed'
            return metadata
        
        def bad_hook(metadata):
            execution_order.append(2)
            raise ValueError("Hook error")
        
        def good_hook2(metadata):
            execution_order.append(3)
            metadata['hook2'] = 'executed'
            return metadata
        
        register_metadata_hook(good_hook1, priority=3)
        register_metadata_hook(bad_hook, priority=2)
        register_metadata_hook(good_hook2, priority=1)
        
        result = execute_metadata_hooks({'agent': 'Test'})
        
        # All hooks should have been attempted
        assert execution_order == [1, 2, 3]
        
        # Good hooks should have modified metadata
        assert result['hook1'] == 'executed'
        assert result['hook2'] == 'executed'
        assert result['agent'] == 'Test'
    
    def test_hook_returns_non_dict(self):
        """Test that hook returning non-dict is handled gracefully."""
        def bad_hook(metadata):
            return "not a dict"
        
        def good_hook(metadata):
            metadata['good'] = 'value'
            return metadata
        
        register_metadata_hook(bad_hook, priority=2)
        register_metadata_hook(good_hook, priority=1)
        
        result = execute_metadata_hooks({'agent': 'Test'})
        
        # Bad hook should be skipped, good hook should execute
        assert result['good'] == 'value'
        assert result['agent'] == 'Test'


class TestHookUseCases:
    """Test suite for common hook use cases."""
    
    def setup_method(self):
        """Clear hooks before each test."""
        clear_metadata_hooks()
    
    def teardown_method(self):
        """Clear hooks after each test."""
        clear_metadata_hooks()
    
    def test_add_environment_hook(self):
        """Test hook that adds environment information."""
        def add_environment(metadata):
            metadata['environment'] = 'production'
            metadata['region'] = 'us-east-1'
            return metadata
        
        register_metadata_hook(add_environment)
        
        result = execute_metadata_hooks({'agent': 'Test'})
        assert result['environment'] == 'production'
        assert result['region'] == 'us-east-1'
    
    def test_validation_hook(self):
        """Test hook that validates required fields."""
        def validate_required(metadata):
            if 'organization_id' not in metadata:
                raise ValueError("organization_id is required")
            return metadata

        register_metadata_hook(validate_required)

        # In real usage, the exception is caught and logged by execute_metadata_hooks
        # The hook should be skipped and execution should continue
        result = execute_metadata_hooks({'agent': 'Test'})

        # The hook should have been skipped due to the exception
        # Original metadata should be returned
        assert result == {'agent': 'Test'}
    
    def test_enrichment_hook(self):
        """Test hook that enriches metadata with additional context."""
        def enrich_metadata(metadata):
            # Simulate looking up additional info based on agent
            if metadata.get('agent') == 'Lead Analyst':
                metadata['team'] = 'Research'
                metadata['cost_center'] = 'R&D'
            return metadata
        
        register_metadata_hook(enrich_metadata)
        
        result = execute_metadata_hooks({'agent': 'Lead Analyst'})
        assert result['team'] == 'Research'
        assert result['cost_center'] == 'R&D'

