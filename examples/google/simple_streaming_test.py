"""
Revenium Google AI Middleware - Streaming Test Examples

This script demonstrates streaming functionality with the Revenium middleware for both
Google AI SDK (Gemini Developer API) and Vertex AI SDK.

Requirements:
- REVENIUM_METERING_API_KEY environment variable
- For Google AI: GOOGLE_API_KEY
- For Vertex AI: GOOGLE_CLOUD_PROJECT and authentication

Usage:
    python simple_streaming_test.py                    # Test Google AI SDK (default)
    python simple_streaming_test.py --provider google-ai    # Test Google AI SDK explicitly
    python simple_streaming_test.py --provider vertex-ai    # Test Vertex AI SDK
    python simple_streaming_test.py --help                  # Show help
"""

import argparse
import os
import sys
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    print(" Loaded environment variables from .env file")
except ImportError:
    print(" Tip: Install python-dotenv for .env file support")
    print("   pip install python-dotenv")


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Revenium Google AI Middleware Streaming Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Test Google AI SDK streaming (default)
  %(prog)s --provider google-ai      # Test Google AI SDK streaming explicitly
  %(prog)s --provider vertex-ai      # Test Vertex AI SDK streaming

Environment Setup:

For Google AI SDK (Gemini Developer API):
  export GOOGLE_API_KEY=your_google_api_key
  export REVENIUM_METERING_API_KEY=your_revenium_key

For Vertex AI SDK (Google Cloud):
  export GOOGLE_CLOUD_PROJECT=your_project_id
  export GOOGLE_CLOUD_LOCATION=us-central1  # optional, defaults to us-central1
  export REVENIUM_METERING_API_KEY=your_revenium_key
  # Ensure Google Cloud authentication is configured (gcloud auth application-default login)

Expected Outputs:
   PASS: Successful streaming API calls with real-time token counting and Revenium metering
   FAIL: Missing environment variables or authentication issues

For more information, visit: https://github.com/revenium/revenium-python-sdk
        """,
    )

    parser.add_argument(
        "--provider",
        choices=["google-ai", "vertex-ai"],
        default="google-ai",
        help="Google AI service to test (default: google-ai)",
    )

    return parser


def validate_environment(provider):
    """Validate required environment variables for the selected provider.

    Args:
        provider (str): The provider to validate ('google-ai' or 'vertex-ai')

    Returns:
        bool: True if environment is valid, False otherwise
    """
    revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

    if not revenium_key:
        print("  REVENIUM_METERING_API_KEY not found")
        print("   Set: export REVENIUM_METERING_API_KEY=your_revenium_key")
        print("   Metering will fail without this key")
        print()

    if provider == "google-ai":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print(" Missing required environment variable for Google AI SDK")
            print("   GOOGLE_API_KEY not found")
            print()
            print(" Setup Instructions:")
            print("   1. Get your API key from: https://aistudio.google.com/app/apikey")
            print("   2. Set the environment variable:")
            print("      export GOOGLE_API_KEY=your_google_api_key")
            print("   3. Run the test again")
            return False

        print(f" Google API Key: {google_api_key[:10]}...")

    elif provider == "vertex-ai":
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            print(" Missing required environment variable for Vertex AI SDK")
            print("   GOOGLE_CLOUD_PROJECT not found")
            print()
            print(" Setup Instructions:")
            print("   1. Create or select a Google Cloud project")
            print("   2. Enable the Vertex AI API in your project")
            print("   3. Set up authentication:")
            print("      gcloud auth application-default login")
            print("   4. Set the environment variables:")
            print("      export GOOGLE_CLOUD_PROJECT=your_project_id")
            print("      export GOOGLE_CLOUD_LOCATION=us-central1  # optional")
            print("   5. Run the test again")
            return False

        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        print(f" Project ID: {project_id}")
        print(f" Location: {location}")

    if revenium_key:
        print(f" Revenium Key: {revenium_key[:10]}...")

    return True


def simple_streaming_test() -> bool:
    """Simple streaming test with Google AI.

    Returns:
        bool: True if test passed, False if failed
    """
    print("\nGoogle AI SDK - Streaming Test")
    print("=" * 50)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

    if revenium_key:
        print(f" Revenium Key: {revenium_key[:8]}...")

    try:
        # Import the middleware (this activates it)
        print("\n Importing Revenium middleware...")
        import revenium_middleware.google  # noqa: F401 - imported for side effects

        print(" Middleware imported and activated")

        # Import Google AI SDK
        print(" Importing Google AI SDK...")
        from google import genai

        print(" Google AI SDK imported")

        # Create client
        print("Creating Google AI client...")
        client = genai.Client(api_key=google_api_key)
        print(" Gemini Developer API client created")

        # Make streaming API call with Revenium metadata
        print(" Making streaming API call...")
        print(" Response (streaming):")
        print("-" * 30)

        start_time = time.time()
        full_response = ""
        chunk_count = 0

        stream = client.models.generate_content_stream(
            model="gemini-2.0-flash-001",
            contents="Write a short 3-sentence story about a robot learning to paint. Stream the response.",
            usage_metadata={
                "trace_id": "streaming-test-001",
                "task_type": "creative-writing",
                "subscriber": {
                    "email": "test@example.com",
                },
                "organization_name": "test-org",
                "agent": "streaming-test-bot",
            },
        )

        # Process streaming response
        for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
                chunk_count += 1

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "-" * 30)
        print(f" Streaming completed!")
        print(f" Received {chunk_count} chunks in {duration:.2f} seconds")
        print(f" Total response length: {len(full_response)} characters")

        # Note: For streaming responses, usage metadata might not be immediately available
        # The middleware should still capture and meter the usage when the stream completes
        print(" Usage will be metered when stream completes")

        print("\n Streaming test completed successfully!")
        print("   Check your Revenium dashboard for metering data")
        return True

    except Exception as e:
        print(f" Streaming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def simple_vertex_ai_streaming_test() -> bool:
    """Simple streaming test with Vertex AI SDK.

    Returns:
        bool: True if test passed, False if failed
    """
    print("\nVertex AI SDK - Streaming Test")
    print("=" * 50)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    try:
        # Import the middleware (this activates it)
        print("\n Importing Revenium middleware...")
        import revenium_middleware.google  # noqa: F401 - imported for side effects

        print(" Middleware imported and activated")

        # Import Vertex AI SDK
        print(" Importing Vertex AI SDK...")
        import vertexai
        from vertexai.generative_models import GenerativeModel

        print(" Vertex AI SDK imported")

        # Initialize Vertex AI
        print("Initializing Vertex AI...")
        vertexai.init(project=project_id, location=location)
        print(" Vertex AI initialized")

        # Create model
        print(" Creating Vertex AI model...")
        model = GenerativeModel("gemini-2.0-flash-001")
        print(" Model created")

        # Make streaming API call
        print(" Making streaming API call...")
        print(" Response (streaming):")
        print("-" * 30)

        start_time = time.time()
        full_response = ""
        chunk_count = 0

        # Generate streaming content
        stream = model.generate_content(
            "Write a short 3-sentence story about a robot learning to paint. Stream the response.",
            stream=True,  # Enable streaming for Vertex AI
        )

        # Process streaming response
        for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
                chunk_count += 1

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "-" * 30)
        print(f" Streaming completed!")
        print(f" Received {chunk_count} chunks in {duration:.2f} seconds")
        print(f" Total response length: {len(full_response)} characters")

        print(" Usage will be metered when stream completes")

        print("\n Vertex AI streaming test completed successfully!")
        print("   Check your Revenium dashboard for metering data")
        return True

    except Exception as e:
        print(f" Vertex AI streaming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Run streaming tests for the selected provider.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    provider = args.provider

    print("Revenium Google AI Middleware - Streaming Test Suite")
    print(f"Testing: {provider.upper().replace('-', ' ')} SDK Streaming")
    print("=" * 70)

    # Validate environment for selected provider
    if not validate_environment(provider):
        print("\n Environment validation failed")
        print("   Please fix the configuration issues above and try again")
        return 1

    print()  # Add spacing after environment validation

    success = False

    if provider == "google-ai":
        print(" GOOGLE AI SDK STREAMING TEST")
        print("=" * 70)

        try:
            import google.genai  # noqa: F401 - checking availability

            print(" Google AI SDK detected")
            success = simple_streaming_test()
        except ImportError:
            print(" Google AI SDK not available")
            print('   Install with: pip install "revenium-python-sdk[google-genai]"')

    elif provider == "vertex-ai":
        print("  VERTEX AI SDK STREAMING TEST")
        print("=" * 70)

        try:
            import vertexai  # noqa: F401 - checking availability

            print(" Vertex AI SDK detected")
            success = simple_vertex_ai_streaming_test()
        except ImportError:
            print(" Vertex AI SDK not available")
            print('   Install with: pip install "revenium-python-sdk[google-vertex]"')

    # Summary
    print("\n" + "=" * 70)
    print(" STREAMING TEST RESULTS")
    print("=" * 70)

    if success:
        print(" PASS: Streaming test successful!")
        print(" Check your Revenium dashboard for streaming usage data")

        # Only show debug tip if debug logging is not already enabled
        try:
            from revenium_middleware.google.common.utils import is_debug_logging_enabled

            if not is_debug_logging_enabled():
                print(" Tip: Enable debug logging with REVENIUM_LOG_LEVEL=DEBUG")
        except ImportError:
            # Fallback if import fails
            import os

            if os.getenv("REVENIUM_LOG_LEVEL", "INFO").upper() != "DEBUG":
                print(" Tip: Enable debug logging with REVENIUM_LOG_LEVEL=DEBUG")

        return 0
    else:
        print(" FAIL: Streaming test failed")
        print("   Check your configuration and API keys")
        return 1


if __name__ == "__main__":
    sys.exit(main())
