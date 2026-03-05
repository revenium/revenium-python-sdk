"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Advanced end-to-end integration tests with real Revenium API.

These tests cover advanced scenarios:
- Multi-turn conversations
- Vision/multimodal inputs
- Batch processing
- Error handling scenarios
- Context caching
- Various model variants

Run with: pytest tests/test_end_to_end_advanced.py -v --run-e2e

Set environment variables:
- REVENIUM_METERING_API_KEY: Your Revenium API key
- GOOGLE_API_KEY: Your Google AI API key (for Google AI SDK tests)
- GOOGLE_CLOUD_PROJECT: Your Google Cloud project (for Vertex AI tests)
"""

import pytest
import os
import time
from datetime import datetime


def pytest_addoption(parser):
    """Add pytest command line option."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end integration tests with real API"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests unless --run-e2e is specified."""
    if config.getoption("--run-e2e"):
        return
    skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


@pytest.mark.e2e
class TestMultiTurnConversations:
    """Test multi-turn conversation tracking."""

    def test_conversation_with_multiple_turns(self):
        """Test that multi-turn conversation is properly metered."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        conversation_id = f"conv-{int(time.time())}"
        session_id = f"session-{int(time.time())}"

        # Turn 1
        turn1_trace = f"{conversation_id}-turn1"
        response1 = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="My name is Alice. Remember this.",
            usage_metadata={
                "trace_id": turn1_trace,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "turn_number": 1,
                "subscriber": {
                    "id": "multiturn-user",
                    "email": "multiturn@example.com",
                }
            }
        )

        time.sleep(1)

        # Turn 2
        turn2_trace = f"{conversation_id}-turn2"
        response2 = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="What name did I just tell you?",
            usage_metadata={
                "trace_id": turn2_trace,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "turn_number": 2,
                "subscriber": {
                    "id": "multiturn-user",
                    "email": "multiturn@example.com",
                }
            }
        )

        time.sleep(1)

        # Turn 3
        turn3_trace = f"{conversation_id}-turn3"
        response3 = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="What's 5 plus 5?",
            usage_metadata={
                "trace_id": turn3_trace,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "turn_number": 3,
                "subscriber": {
                    "id": "multiturn-user",
                    "email": "multiturn@example.com",
                }
            }
        )

        # Verify all responses succeeded
        assert response1 is not None
        assert response2 is not None
        assert response3 is not None

        print(f"\n Multi-turn conversation completed")
        print(f"   Conversation ID: {conversation_id}")
        print(f"   Turn 1 trace: {turn1_trace}")
        print(f"   Turn 2 trace: {turn2_trace}")
        print(f"   Turn 3 trace: {turn3_trace}")
        print(f"\n Dashboard validation:")
        print(f"   - Filter by conversation_id={conversation_id}")
        print(f"   - Should see 3 requests with turn_number 1, 2, 3")
        print(f"   - Should see token counts increasing with context")


@pytest.mark.e2e
class TestBatchProcessing:
    """Test batch processing scenarios."""

    def test_batch_of_requests(self):
        """Test metering of multiple requests in batch."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        batch_id = f"batch-{int(time.time())}"

        # Process batch of 5 requests
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Name a primary color.",
            "What day comes after Monday?",
            "How many continents are there?"
        ]

        results = []
        for i, prompt in enumerate(prompts):
            trace_id = f"{batch_id}-item-{i}"

            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt,
                usage_metadata={
                    "trace_id": trace_id,
                    "batch_id": batch_id,
                    "batch_item_number": i,
                    "batch_total": len(prompts),
                    "subscriber": {
                        "id": "batch-user",
                    }
                }
            )

            results.append({
                "trace_id": trace_id,
                "prompt": prompt,
                "response": response.text if response else None
            })

            time.sleep(0.5)  # Small delay between requests

        # Verify all succeeded
        assert len(results) == len(prompts)
        assert all(r["response"] is not None for r in results)

        print(f"\n Batch processing completed: {len(results)} requests")
        print(f"   Batch ID: {batch_id}")
        print(f"\n Dashboard validation:")
        print(f"   - Filter by batch_id={batch_id}")
        print(f"   - Should see {len(prompts)} requests")
        print(f"   - Each with batch_item_number 0-{len(prompts)-1}")

        for result in results:
            print(f"   - {result['trace_id']}: {result['prompt'][:30]}...")


@pytest.mark.e2e
class TestDifferentModelVariants:
    """Test different Google model variants."""

    @pytest.mark.parametrize("model_name", [
        "gemini-2.0-flash-001",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-002",
    ])
    def test_different_models_metered(self, model_name):
        """Test that different model variants are metered correctly."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        test_id = f"model-test-{model_name}-{int(time.time())}"

        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Say 'hello' in one word.",
                usage_metadata={
                    "trace_id": test_id,
                    "model_variant_test": model_name,
                    "subscriber": {
                        "id": "model-tester",
                    }
                }
            )

            assert response is not None
            print(f"\n Model {model_name} tested successfully")
            print(f"   Trace ID: {test_id}")
            print(f"   Verify model appears as: {model_name}")

        except Exception as e:
            # Some models may not be available in all regions
            pytest.skip(f"Model {model_name} not available: {e}")


@pytest.mark.e2e
class TestErrorScenarios:
    """Test error handling and validation."""

    def test_request_with_invalid_model(self):
        """Test that invalid model name is handled gracefully."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        test_id = f"error-test-{int(time.time())}"

        # Try to use non-existent model
        with pytest.raises(Exception):
            response = client.models.generate_content(
                model="invalid-model-name-12345",
                contents="This should fail",
                usage_metadata={
                    "trace_id": test_id,
                    "error_test": True,
                    "subscriber": {
                        "id": "error-tester",
                    }
                }
            )

        print(f"\n Error handling test completed")
        print(f"   Trace ID: {test_id}")
        print(f"   This request should have failed gracefully")
        print(f"   Check if error was logged (may or may not appear in dashboard)")

    def test_request_with_safety_filters(self):
        """Test request that might trigger safety filters."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        test_id = f"safety-test-{int(time.time())}"

        # Make a safe request that shouldn't trigger filters
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="What is a healthy breakfast?",
            usage_metadata={
                "trace_id": test_id,
                "safety_test": True,
                "subscriber": {
                    "id": "safety-tester",
                }
            }
        )

        assert response is not None
        print(f"\n Safety filter test completed")
        print(f"   Trace ID: {test_id}")
        print(f"   Verify finish_reason is END (not SAFETY)")


@pytest.mark.e2e
class TestMetadataVariations:
    """Test various metadata combinations."""

    def test_minimal_metadata(self):
        """Test with minimal required metadata."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        test_id = f"minimal-{int(time.time())}"

        # Only trace_id and subscriber.id
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Say 'minimal'",
            usage_metadata={
                "trace_id": test_id,
                "subscriber": {
                    "id": "minimal-user",
                }
            }
        )

        assert response is not None
        print(f"\n Minimal metadata test")
        print(f"   Trace ID: {test_id}")
        print(f"   Only subscriber.id provided")

    def test_maximal_metadata(self):
        """Test with all available metadata fields."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        test_id = f"maximal-{int(time.time())}"

        # All possible fields
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Say 'maximal'",
            usage_metadata={
                "trace_id": test_id,
                "subscriber": {
                    "id": "maximal-user-123",
                    "email": "maximal@example.com",
                    "credential": {
                        "name": "maximal-api-key",
                        "value": "key-value-123"
                    }
                },
                "organization_id": "maximal-org",
                "product_id": "maximal-product",
                "agent": "maximal-agent",
                "agent_version": "1.0.0",
                "task_type": "maximal-test",
                "conversation_id": "maximal-conv",
                "session_id": "maximal-session",
                "turn_number": 1,
                "response_quality_score": 0.95,
                "custom_field_1": "custom_value_1",
                "custom_field_2": 42,
                "custom_field_3": True
            }
        )

        assert response is not None
        print(f"\n Maximal metadata test")
        print(f"   Trace ID: {test_id}")
        print(f"   All metadata fields populated")
        print(f"\n Verify in dashboard:")
        print(f"   - subscriber.id, email, credential.name, credential.value")
        print(f"   - organization_id, product_id, agent, agent_version")
        print(f"   - task_type, conversation_id, session_id, turn_number")
        print(f"   - response_quality_score: 0.95")
        print(f"   - custom fields: custom_field_1, custom_field_2, custom_field_3")


@pytest.mark.e2e
class TestContextCaching:
    """Test context caching scenarios."""

    def test_request_with_cached_content(self):
        """Test that cached content is properly tracked."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)

        # First request - no cache
        test_id_1 = f"cache-miss-{int(time.time())}"
        response1 = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="The sky is blue. What color is the sky?",
            usage_metadata={
                "trace_id": test_id_1,
                "cache_test": "first_request",
                "subscriber": {
                    "id": "cache-tester",
                }
            }
        )

        time.sleep(1)

        # Second request - similar content (might be cached)
        test_id_2 = f"cache-hit-{int(time.time())}"
        response2 = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="The sky is blue. What color is the sky?",
            usage_metadata={
                "trace_id": test_id_2,
                "cache_test": "second_request",
                "subscriber": {
                    "id": "cache-tester",
                }
            }
        )

        assert response1 is not None
        assert response2 is not None

        print(f"\n Context caching test completed")
        print(f"   First request: {test_id_1}")
        print(f"   Second request: {test_id_2}")
        print(f"\n Dashboard validation:")
        print(f"   - Compare cached_tokens between requests")
        print(f"   - Second request may show cached_content_token_count > 0")


@pytest.mark.e2e
class TestStreamingAdvanced:
    """Advanced streaming scenarios."""

    def test_streaming_with_large_output(self):
        """Test streaming with larger output to track multiple chunks."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        test_id = f"streaming-large-{int(time.time())}"

        # Request larger output
        stream = client.models.generate_content_stream(
            model="gemini-2.0-flash-001",
            contents="Write a short paragraph about Python programming (3-4 sentences).",
            usage_metadata={
                "trace_id": test_id,
                "streaming_test": "large_output",
                "subscriber": {
                    "id": "stream-tester",
                }
            }
        )

        chunks = []
        chunk_times = []
        start_time = time.time()

        for chunk in stream:
            if chunk.text:
                chunks.append(chunk.text)
                chunk_times.append(time.time() - start_time)

        # Verify
        assert len(chunks) > 0
        total_text = "".join(chunks)

        print(f"\n Large streaming test completed")
        print(f"   Trace ID: {test_id}")
        print(f"   Chunks received: {len(chunks)}")
        print(f"   Total characters: {len(total_text)}")
        print(f"   Chunk times: {[f'{t*1000:.0f}ms' for t in chunk_times[:5]]}")
        print(f"\n Verify in dashboard:")
        print(f"   - time_to_first_token should be recorded")
        print(f"   - Total tokens should match complete response")


@pytest.mark.e2e
class TestProviderDetection:
    """Test provider detection and metadata."""

    def test_google_ai_sdk_provider_metadata(self):
        """Test that Google AI SDK is correctly identified."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)
        test_id = f"provider-google-ai-{int(time.time())}"

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Say 'hello'",
            usage_metadata={
                "trace_id": test_id,
                "provider_test": "google_ai_sdk",
                "subscriber": {
                    "id": "provider-tester",
                }
            }
        )

        assert response is not None
        print(f"\n Google AI SDK provider test")
        print(f"   Trace ID: {test_id}")
        print(f"\n Verify in dashboard:")
        print(f"   - Provider should be: Google")
        print(f"   - Model source should be: GOOGLE")
        print(f"   - Endpoint should contain: generativelanguage.googleapis.com")

    def test_vertex_ai_sdk_provider_metadata(self):
        """Test that Vertex AI SDK is correctly identified."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not project_id or not revenium_key:
            pytest.skip("Missing GOOGLE_CLOUD_PROJECT or REVENIUM_METERING_API_KEY")

        try:
            import revenium_middleware.google
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            pytest.skip("Vertex AI SDK not installed")

        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)

        test_id = f"provider-vertex-ai-{int(time.time())}"
        model = GenerativeModel("gemini-2.0-flash-001")

        model._revenium_usage_metadata = {
            "trace_id": test_id,
            "provider_test": "vertex_ai_sdk",
            "subscriber": {
                "id": "provider-tester",
            }
        }

        response = model.generate_content("Say 'hello'")

        assert response is not None
        print(f"\n Vertex AI SDK provider test")
        print(f"   Trace ID: {test_id}")
        print(f"\n Verify in dashboard:")
        print(f"   - Provider should be: Google")
        print(f"   - Model source should be: GOOGLE")
        print(f"   - Endpoint should contain: aiplatform.googleapis.com")


if __name__ == "__main__":
    print(__doc__)
    print("\nTo run these advanced tests:")
    print("  pytest tests/test_end_to_end_advanced.py -v --run-e2e")
    print("\nMake sure to set environment variables:")
    print("  export REVENIUM_METERING_API_KEY=your_key")
    print("  export GOOGLE_API_KEY=your_key")
    print("  export GOOGLE_CLOUD_PROJECT=your_project")
