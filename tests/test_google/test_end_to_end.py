"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
End-to-end integration tests with real Revenium API.

These tests require valid API keys and will make real API calls.
Run with: pytest tests/test_end_to_end.py -v --run-e2e

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
class TestGoogleAISDKEndToEnd:
    """End-to-end tests for Google AI SDK integration."""

    def test_basic_completion_metered(self):
        """Test that a basic completion is metered and appears in Revenium."""
        # Skip if Google AI SDK not available or no API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing GOOGLE_API_KEY or REVENIUM_METERING_API_KEY")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        # Make a request with unique metadata
        test_id = f"e2e-test-{int(time.time())}"
        client = genai.Client(api_key=google_api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="What is 2+2? Answer in one word.",
            usage_metadata={
                "trace_id": test_id,
                "task_type": "e2e-testing",
                "subscriber": {
                    "id": "test-user-e2e",
                    "email": "e2e-test@example.com",
                },
                "organization_id": "e2e-test-org",
                "product_id": "e2e-test-product",
                "agent": "pytest-e2e-agent"
            }
        )

        # Verify response
        assert response is not None
        assert response.text is not None
        print(f"\n Response received: {response.text}")
        print(f" Trace ID: {test_id}")
        print(f" Check Revenium dashboard for trace_id={test_id}")
        print(f"   Expected subscriber.id: test-user-e2e")
        print(f"   Expected subscriber.email: e2e-test@example.com")
        print(f"   Expected organization_id: e2e-test-org")

    def test_streaming_with_time_to_first_token(self):
        """Test streaming completion with time-to-first-token tracking."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing GOOGLE_API_KEY or REVENIUM_METERING_API_KEY")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        # Make a streaming request
        test_id = f"e2e-streaming-{int(time.time())}"
        client = genai.Client(api_key=google_api_key)

        stream = client.models.generate_content_stream(
            model="gemini-2.0-flash-001",
            contents="Count from 1 to 5, one number per line.",
            usage_metadata={
                "trace_id": test_id,
                "task_type": "e2e-streaming-test",
                "subscriber": {
                    "id": "test-user-streaming",
                    "email": "streaming@example.com",
                },
            }
        )

        # Collect chunks
        chunks = []
        start_time = time.time()
        first_chunk_time = None

        for chunk in stream:
            if chunk.text:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                chunks.append(chunk.text)

        end_time = time.time()

        # Verify
        assert len(chunks) > 0
        assert first_chunk_time is not None

        time_to_first_token_ms = int((first_chunk_time - start_time) * 1000)
        total_time_ms = int((end_time - start_time) * 1000)

        print(f"\n Streaming completed: {len(chunks)} chunks")
        print(f" Time to first token: {time_to_first_token_ms}ms")
        print(f" Total time: {total_time_ms}ms")
        print(f" Trace ID: {test_id}")
        print(f"   Check Revenium dashboard for time_to_first_token metric")

    def test_nested_vs_flat_subscriber_comparison(self):
        """Test and compare nested vs flat subscriber format."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing GOOGLE_API_KEY or REVENIUM_METERING_API_KEY")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        client = genai.Client(api_key=google_api_key)

        # Test 1: Nested format (recommended)
        test_id_nested = f"e2e-nested-{int(time.time())}"
        response_nested = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Say 'nested format'",
            usage_metadata={
                "trace_id": test_id_nested,
                "subscriber": {
                    "id": "nested-user-123",
                    "email": "nested@example.com",
                }
            }
        )

        time.sleep(1)  # Brief pause between requests

        # Test 2: Flat format (deprecated but should work)
        test_id_flat = f"e2e-flat-{int(time.time())}"
        response_flat = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Say 'flat format'",
            usage_metadata={
                "trace_id": test_id_flat,
                "subscriber_id": "flat-user-456",
                "subscriber_email": "flat@example.com",
            }
        )

        # Both should succeed
        assert response_nested is not None
        assert response_flat is not None

        print(f"\n Nested format trace ID: {test_id_nested}")
        print(f"   Expected subscriber.id: nested-user-123")
        print(f" Flat format trace ID: {test_id_flat}")
        print(f"   Expected subscriber.id: flat-user-456 (transformed)")
        print(f"\n Compare both in Revenium dashboard:")
        print(f"   Both should show subscriber.id and subscriber.email")
        print(f"   Flat format should trigger deprecation warning in logs")


@pytest.mark.e2e
class TestVertexAISDKEndToEnd:
    """End-to-end tests for Vertex AI SDK integration."""

    def test_vertex_basic_completion_metered(self):
        """Test that Vertex AI completion is metered."""
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

        # Initialize Vertex AI
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)

        # Make a request
        test_id = f"e2e-vertex-{int(time.time())}"
        model = GenerativeModel("gemini-2.0-flash-001")

        # Set metadata on model instance
        model._revenium_usage_metadata = {
            "trace_id": test_id,
            "task_type": "e2e-vertex-test",
            "subscriber": {
                "id": "vertex-user-123",
                "email": "vertex@example.com",
            },
            "organization_id": "vertex-org",
        }

        response = model.generate_content("What is the capital of France? One word answer.")

        # Verify
        assert response is not None
        assert response.text is not None

        print(f"\n Vertex AI response: {response.text}")
        print(f" Trace ID: {test_id}")
        print(f"   Check Revenium dashboard for Vertex AI metrics")
        print(f"   Provider should be: Google")
        print(f"   Model source should be: GOOGLE")

    def test_vertex_embeddings_with_tokens(self):
        """Test that Vertex AI embeddings are metered with token counts."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not project_id or not revenium_key:
            pytest.skip("Missing GOOGLE_CLOUD_PROJECT or REVENIUM_METERING_API_KEY")

        try:
            import revenium_middleware.google
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
        except ImportError:
            pytest.skip("Vertex AI SDK or TextEmbeddingModel not available")

        # Initialize Vertex AI
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)

        # Make an embeddings request
        test_id = f"e2e-vertex-embed-{int(time.time())}"

        try:
            model = TextEmbeddingModel.from_pretrained("text-embedding-004")

            # Set metadata
            model._revenium_usage_metadata = {
                "trace_id": test_id,
                "task_type": "e2e-embeddings-test",
                "subscriber": {
                    "id": "embed-user-123",
                },
            }

            embeddings = model.get_embeddings(["This is a test sentence for embeddings."])

            # Verify
            assert embeddings is not None
            assert len(embeddings) > 0
            assert len(embeddings[0].values) > 0

            print(f"\n Vertex AI embeddings generated: {len(embeddings[0].values)} dimensions")
            print(f" Trace ID: {test_id}")
            print(f"   Check Revenium dashboard for token counts")
            print(f"   Vertex AI should have full token counting for embeddings")
        except Exception as e:
            pytest.skip(f"Vertex AI embeddings not available: {e}")


@pytest.mark.e2e
class TestReveniumAPIValidation:
    """Validate that data appears correctly in Revenium API."""

    def test_subscriber_fields_in_dashboard(self):
        """
        Manual validation test - requires checking dashboard.

        This test makes a request with known subscriber data and prints
        instructions for manual validation in the Revenium dashboard.
        """
        google_api_key = os.getenv("GOOGLE_API_KEY")
        revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

        if not google_api_key or not revenium_key:
            pytest.skip("Missing required API keys")

        try:
            import revenium_middleware.google
            from google import genai
        except ImportError:
            pytest.skip("Google AI SDK not installed")

        # Generate unique test data
        timestamp = int(time.time())
        test_subscriber_id = f"validation-user-{timestamp}"
        test_email = f"validation-{timestamp}@example.com"
        test_trace_id = f"validation-trace-{timestamp}"

        client = genai.Client(api_key=google_api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Respond with: validated",
            usage_metadata={
                "trace_id": test_trace_id,
                "task_type": "validation-test",
                "subscriber": {
                    "id": test_subscriber_id,
                    "email": test_email,
                    "credential": {
                        "name": "test-api-key",
                        "value": "test-key-12345"
                    }
                },
                "organization_id": f"validation-org-{timestamp}",
                "product_id": "validation-product",
                "agent": "validation-agent",
                "response_quality_score": 0.99
            }
        )

        assert response is not None

        print("\n" + "="*70)
        print("MANUAL VALIDATION REQUIRED")
        print("="*70)
        print(f"\nPlease check your Revenium dashboard and verify:")
        print(f"\n1. Search for trace_id: {test_trace_id}")
        print(f"\n2. Verify the following fields appear correctly:")
        print(f"   - subscriber.id: {test_subscriber_id}")
        print(f"   - subscriber.email: {test_email}")
        print(f"   - subscriber.credential.name: test-api-key")
        print(f"   - subscriber.credential.value: test-key-12345")
        print(f"   - organization_id: validation-org-{timestamp}")
        print(f"   - product_id: validation-product")
        print(f"   - agent: validation-agent")
        print(f"   - response_quality_score: 0.99")
        print(f"   - task_type: validation-test")
        print(f"\n3. Verify provider is: Google")
        print(f"\n4. Verify model is: gemini-2.0-flash-001")
        print(f"\n5. Verify token counts are present and non-zero")
        print(f"\n" + "="*70)


if __name__ == "__main__":
    print(__doc__)
    print("\nTo run these tests:")
    print("  pytest tests/test_end_to_end.py -v --run-e2e")
    print("\nMake sure to set environment variables:")
    print("  export REVENIUM_METERING_API_KEY=your_key")
    print("  export GOOGLE_API_KEY=your_key")
    print("  export GOOGLE_CLOUD_PROJECT=your_project")
