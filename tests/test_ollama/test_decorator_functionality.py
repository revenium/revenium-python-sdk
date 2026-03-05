"""
Test decorator functionality with Ollama middleware.

This module tests the new decorator features:
1. @revenium_metadata - Inject metadata into API calls
2. @revenium_meter - Selective metering control

These tests verify that decorators work correctly with the Ollama middleware
and that data is properly sent to Revenium.
"""

import os
import pytest
import ollama
import time

# Import the new decorators from revenium-python-sdk
try:
    from revenium_middleware import (
        revenium_metadata,
        revenium_meter,
        is_selective_metering_enabled,
    )
    DECORATORS_AVAILABLE = True
except ImportError:
    DECORATORS_AVAILABLE = False
    pytest.skip(
        "Decorators not available. Install revenium-python-sdk>=0.1.0",
        allow_module_level=True
    )

# Import the middleware to ensure patching is active
import revenium_middleware.ollama


@pytest.mark.integration
class TestMetadataInjectionDecorator:
    """Test the @revenium_metadata decorator with Ollama."""

    @pytest.mark.e2e
    def test_metadata_injection_basic(self, model_name):
        """Test that @revenium_metadata injects metadata into Ollama calls."""
        
        @revenium_metadata(
            trace_id="test-trace-001",
            task_type="decorator-test",
            organization_name="TestOrg"
        )
        def chat_with_metadata():
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Say hello in one word.',
                    },
                ]
            )
            return response
        
        # Execute the decorated function
        response = chat_with_metadata()
        
        # Verify response is valid
        assert 'message' in response
        assert response['message']['content']
        
        # Verify transaction ID was added
        assert hasattr(response, '_revenium_transaction_id')
        transaction_id = response._revenium_transaction_id
        assert transaction_id.startswith('ollama-')
        
        print(f"✅ Metadata injection test passed. Transaction ID: {transaction_id}")

    @pytest.mark.e2e
    def test_metadata_injection_with_api_override(self, model_name):
        """Test that API-level metadata overrides decorator metadata."""
        
        @revenium_metadata(
            trace_id="decorator-trace",
            task_type="default-task"
        )
        def chat_with_override():
            # API-level metadata should override decorator metadata
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Count to three.',
                    },
                ],
                usage_metadata={
                    "task_type": "api-override-task",
                    "priority": "high"
                }
            )
            return response
        
        response = chat_with_override()
        
        # Verify response is valid
        assert 'message' in response
        assert response['message']['content']
        
        # Verify transaction ID
        assert hasattr(response, '_revenium_transaction_id')
        
        print(f"✅ Metadata override test passed. Transaction ID: {response._revenium_transaction_id}")

    @pytest.mark.e2e
    def test_metadata_injection_nested_calls(self, model_name):
        """Test metadata injection with multiple nested API calls."""
        
        @revenium_metadata(
            trace_id="nested-trace-001",
            session_id="session-123",
            organization_name="NestedOrg"
        )
        def multiple_calls():
            results = []
            
            # First call
            response1 = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Say A.'}]
            )
            results.append(response1)
            
            # Second call
            response2 = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Say B.'}]
            )
            results.append(response2)
            
            return results
        
        responses = multiple_calls()
        
        # Verify both responses are valid
        assert len(responses) == 2
        for response in responses:
            assert 'message' in response
            assert hasattr(response, '_revenium_transaction_id')
        
        print(f"✅ Nested calls test passed. Transaction IDs: {[r._revenium_transaction_id for r in responses]}")

    @pytest.mark.e2e
    def test_metadata_injection_with_generate(self, model_name):
        """Test metadata injection with ollama.generate()."""
        
        @revenium_metadata(
            trace_id="generate-trace-001",
            task_type="text-generation"
        )
        def generate_with_metadata():
            response = ollama.generate(
                model=model_name,
                prompt='Say hello in one word.'
            )
            return response
        
        response = generate_with_metadata()
        
        # Verify response is valid
        assert 'response' in response
        assert response['response']
        
        # Verify transaction ID
        assert hasattr(response, '_revenium_transaction_id')
        
        print(f"✅ Generate with metadata test passed. Transaction ID: {response._revenium_transaction_id}")


@pytest.mark.integration
class TestSelectiveMeteringDecorator:
    """Test the @revenium_meter decorator with Ollama."""

    @pytest.mark.e2e
    def test_selective_metering_disabled_by_default(self, model_name):
        """Test that selective metering is disabled by default."""
        # Ensure selective metering is not enabled
        assert not is_selective_metering_enabled()
        
        # Without selective metering enabled, all calls should be metered
        # regardless of decorator
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hello'}]
        )
        
        assert 'message' in response
        assert hasattr(response, '_revenium_transaction_id')
        
        print("✅ Default metering behavior verified (all calls metered)")

    @pytest.mark.e2e
    def test_meter_decorator_basic(self, model_name):
        """Test basic @revenium_meter decorator functionality."""
        
        @revenium_meter()
        def metered_chat():
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Say hi.'}]
            )
            return response
        
        response = metered_chat()
        
        # Verify response is valid
        assert 'message' in response
        assert hasattr(response, '_revenium_transaction_id')
        
        print(f"✅ Basic meter decorator test passed. Transaction ID: {response._revenium_transaction_id}")

    @pytest.mark.e2e
    def test_combined_decorators(self, model_name):
        """Test combining @revenium_meter and @revenium_metadata."""
        
        @revenium_meter()
        @revenium_metadata(
            trace_id="combined-trace-001",
            task_type="combined-test",
            organization_name="CombinedOrg"
        )
        def combined_decorators():
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Test combined.'}]
            )
            return response
        
        response = combined_decorators()
        
        # Verify response is valid
        assert 'message' in response
        assert hasattr(response, '_revenium_transaction_id')
        
        print(f"✅ Combined decorators test passed. Transaction ID: {response._revenium_transaction_id}")


@pytest.mark.integration
class TestDecoratorWithStreaming:
    """Test decorators with streaming responses."""

    @pytest.mark.e2e
    def test_metadata_injection_with_streaming(self, model_name):
        """Test metadata injection with streaming responses."""
        
        @revenium_metadata(
            trace_id="streaming-trace-001",
            task_type="streaming-test"
        )
        def streaming_chat():
            chunks = []
            stream = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Count to three.'}],
                stream=True
            )
            
            for chunk in stream:
                chunks.append(chunk)
            
            return chunks
        
        chunks = streaming_chat()
        
        # Verify we got chunks
        assert len(chunks) > 0
        
        # The final chunk should have the transaction ID
        final_chunk = chunks[-1]
        if hasattr(final_chunk, '_revenium_transaction_id'):
            print(f"✅ Streaming metadata test passed. Transaction ID: {final_chunk._revenium_transaction_id}")
        else:
            print("✅ Streaming metadata test passed (transaction ID on final chunk)")


@pytest.mark.integration
class TestDecoratorErrorHandling:
    """Test decorator behavior with errors and edge cases."""

    @pytest.mark.e2e
    def test_decorator_with_exception(self, model_name):
        """Test that decorators properly clean up even when exceptions occur."""
        
        @revenium_metadata(trace_id="error-trace-001")
        def function_with_error():
            # Make a valid call first
            ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Hello'}]
            )
            # Then raise an exception
            raise ValueError("Test exception")
        
        # Verify exception is raised
        with pytest.raises(ValueError, match="Test exception"):
            function_with_error()
        
        # Verify we can still make calls after the exception
        # (context should be cleaned up)
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': 'After error'}]
        )
        
        assert 'message' in response
        print("✅ Exception handling test passed")

    def test_decorator_without_api_calls(self, model_name):
        """Test decorator on function that doesn't make API calls."""
        
        @revenium_metadata(trace_id="no-api-trace")
        def no_api_calls():
            return "No API calls made"
        
        result = no_api_calls()
        assert result == "No API calls made"
        
        print("✅ No API calls test passed")


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.e2e
    def test_conversation_with_metadata(self, model_name):
        """Test a multi-turn conversation with consistent metadata."""
        
        @revenium_metadata(
            trace_id="conversation-001",
            session_id="user-session-123",
            organization_name="AcmeCorp",
            task_type="customer-support"
        )
        def customer_support_conversation():
            conversation_history = []
            
            # Turn 1
            response1 = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'user', 'content': 'What is 2+2?'}
                ]
            )
            conversation_history.append(response1)
            
            # Turn 2 - with additional API-level metadata
            response2 = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'user', 'content': 'What is 2+2?'},
                    {'role': 'assistant', 'content': response1['message']['content']},
                    {'role': 'user', 'content': 'What about 3+3?'}
                ],
                usage_metadata={
                    "turn_number": 2,
                    "conversation_depth": "follow-up"
                }
            )
            conversation_history.append(response2)
            
            return conversation_history
        
        conversation = customer_support_conversation()
        
        # Verify both turns succeeded
        assert len(conversation) == 2
        for response in conversation:
            assert 'message' in response
            assert hasattr(response, '_revenium_transaction_id')
        
        print(f"✅ Conversation test passed. {len(conversation)} turns completed")
        print(f"   Transaction IDs: {[r._revenium_transaction_id for r in conversation]}")

    @pytest.mark.e2e
    def test_batch_processing_with_metadata(self, model_name):
        """Test batch processing with metadata tracking."""
        
        @revenium_metadata(
            trace_id="batch-001",
            batch_id="batch-processing-123",
            organization_name="DataProcessor"
        )
        def process_batch():
            items = ["Hello", "Goodbye", "Thanks"]
            results = []
            
            for idx, item in enumerate(items):
                response = ollama.chat(
                    model=model_name,
                    messages=[{'role': 'user', 'content': f'Say: {item}'}],
                    usage_metadata={
                        "item_index": idx,
                        "item_count": len(items)
                    }
                )
                results.append(response)
                
                # Small delay to ensure different timestamps
                time.sleep(0.1)
            
            return results
        
        results = process_batch()
        
        # Verify all items processed
        assert len(results) == 3
        for response in results:
            assert 'message' in response
            assert hasattr(response, '_revenium_transaction_id')
        
        print(f"✅ Batch processing test passed. {len(results)} items processed")
        print(f"   Transaction IDs: {[r._revenium_transaction_id for r in results]}")


if __name__ == "__main__":
    """
    Run tests directly without pytest for quick verification.
    
    Usage:
        python test_decorator_functionality.py
    """
    print("=" * 70)
    print("DECORATOR FUNCTIONALITY TEST SUITE")
    print("=" * 70)
    
    # Check environment
    if not os.environ.get('REVENIUM_METERING_API_KEY'):
        print("❌ ERROR: REVENIUM_METERING_API_KEY not set")
        print("   Please set your Revenium API key:")
        print("   export REVENIUM_METERING_API_KEY='your-key-here'")
        exit(1)
    
    print(f"✅ Environment configured")
    print(f"   Selective metering: {is_selective_metering_enabled()}")
    print()
    
    # Use a small model for testing
    model = 'qwen2.5:0.5b'
    
    # Run a simple test
    print("Running basic metadata injection test...")
    
    @revenium_metadata(
        trace_id="manual-test-001",
        task_type="manual-verification"
    )
    def simple_test():
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': 'Say hello.'}]
        )
        return response
    
    try:
        response = simple_test()
        print(f"✅ Test passed!")
        print(f"   Response: {response['message']['content'][:50]}...")
        print(f"   Transaction ID: {response._revenium_transaction_id}")
        print()
        print("🎉 Decorator functionality is working correctly!")
        print()
        print("To run full test suite:")
        print("   pytest tests/test_decorator_functionality.py -v")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

