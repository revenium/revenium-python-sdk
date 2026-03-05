"""
 Revenium Google AI Middleware - Simple Test Examples

This script demonstrates basic usage of the Revenium middleware with both
Google AI SDK (Gemini Developer API) and Vertex AI SDK.

Features demonstrated:
- Zero-config integration (just import the middleware)
- Enhanced tracking with metadata
- Intelligent provider selection
- Error handling and debugging

Requirements:
- REVENIUM_METERING_API_KEY environment variable
- For Google AI: GOOGLE_API_KEY
- For Vertex AI: GOOGLE_CLOUD_PROJECT and authentication

Usage:
    python simple_test.py                         # Test Google AI SDK (default)
    python simple_test.py --provider google-ai    # Test Google AI SDK explicitly
    python simple_test.py --provider vertex-ai    # Test Vertex AI SDK
    python simple_test.py --help                  # Show help
"""

import argparse
import os
import sys

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
        description=" Revenium Google AI Middleware Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Test Google AI SDK (default)
  %(prog)s --provider google-ai      # Test Google AI SDK explicitly
  %(prog)s --provider vertex-ai      # Test Vertex AI SDK

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
   PASS: Successful API calls with token counting and Revenium metering
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


def google_ai_basic_example():
    """ Zero-Config Integration with Google AI SDK"""
    print("\n Google AI SDK - Basic Example")
    print("=" * 50)

    google_api_key = os.getenv("GOOGLE_API_KEY")

    try:
        # Import middleware first - this activates automatic tracking
        import revenium_middleware.google
        from google import genai

        client = genai.Client(api_key=google_api_key)

        # Simple API call - automatically tracked!
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="What is the meaning of life, the universe and everything?",
        )

        print(f" Response: {response.text[:100]}...")

        # Show token usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            total_tokens = getattr(usage, "total_token_count", "N/A")
            input_tokens = getattr(usage, "prompt_token_count", "N/A")
            output_tokens = getattr(usage, "candidates_token_count", "N/A")
            print(
                f" Tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total"
            )

        print(" Zero-config integration successful!")
        print("   Your usage is automatically tracked in Revenium")
        return True

    except Exception as e:
        print(f" Google AI test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def google_ai_enhanced_example():
    """ Enhanced Tracking with Metadata"""
    print("\n Google AI SDK - Enhanced Tracking Example")
    print("=" * 50)

    google_api_key = os.getenv("GOOGLE_API_KEY")

    try:
        import revenium_middleware.google
        from google import genai

        client = genai.Client(api_key=google_api_key)

        # Enhanced tracking with detailed metadata (using nested subscriber format)
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Analyze this quarterly report for key insights",
            usage_metadata={
                "trace_id": "conv-28a7e9d4",
                "task_type": "document-analysis",
                "subscriber": {
                    "id": "user-12345",
                    "email": "user@example.com",
                },
                "organizationName": "AcmeCorp",
                "subscription_id": "premium-plan",
                "productName": "business-intelligence",
                "agent": "report-analyzer-v2",
            },
        )

        print(f" Response: {response.text[:100]}...")
        print(" Enhanced metadata tracking enabled!")
        print("   Check your Revenium dashboard for detailed analytics")
        return True

    except Exception as e:
        print(f" Enhanced tracking test failed: {e}")
        return False


def vertex_ai_basic_example():
    """ Zero-Config Integration with Vertex AI SDK"""
    print("\n  Vertex AI SDK - Basic Example")
    print("=" * 50)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    try:
        # Import middleware first - this activates automatic tracking
        import revenium_middleware.google
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=project_id, location=location)
        model = GenerativeModel("gemini-2.0-flash-001")

        # Simple API call - automatically tracked with enhanced token counting!
        response = model.generate_content(
            "What is the meaning of life, the universe and everything?"
        )

        print(f" Response: {response.text[:100]}...")

        # Show enhanced token usage (Vertex AI provides complete data)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            total_tokens = getattr(usage, "total_token_count", "N/A")
            input_tokens = getattr(usage, "prompt_token_count", "N/A")
            output_tokens = getattr(usage, "candidates_token_count", "N/A")
            print(
                f" Tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total"
            )

        print(" Zero-config integration successful!")
        print("   Enhanced token counting with Vertex AI SDK")
        return True

    except Exception as e:
        print(f" Vertex AI test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def vertex_ai_enhanced_example():
    """ Enhanced Features with Vertex AI SDK"""
    print("\n Vertex AI SDK - Enhanced Features Example")
    print("=" * 50)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    try:
        import revenium_middleware.google
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=project_id, location="us-central1")
        model = GenerativeModel("gemini-2.0-flash-001")

        # Set metadata on model instance for automatic tracking
        model._revenium_usage_metadata = {
            "trace_id": "conv-28a7e9d4",
            "task_type": "document-analysis",
            "organizationName": "AcmeCorp",
            "productName": "business-intelligence",
        }

        response = model.generate_content(
            "Analyze this quarterly report for key insights"
        )

        print(f" Response: {response.text[:100]}...")
        print(" Full token counting including embeddings!")
        print(" Enhanced metadata tracking enabled!")
        return True

    except Exception as e:
        print(f" Enhanced features test failed: {e}")
        return False


def main():
    """Run examples for the selected provider"""
    parser = create_argument_parser()
    args = parser.parse_args()

    provider = args.provider

    print(" Revenium Google AI Middleware - Test Suite")
    print(f"Testing: {provider.upper().replace('-', ' ')} SDK")
    print("=" * 70)

    # Validate environment for selected provider
    if not validate_environment(provider):
        print("\n Environment validation failed")
        print("   Please fix the configuration issues above and try again")
        return 1

    print()  # Add spacing after environment validation

    results = {}

    if provider == "google-ai":
        # Google AI SDK Examples
        print(" GOOGLE AI SDK EXAMPLES")
        print("=" * 70)

        try:
            import google.genai

            print(" Google AI SDK detected")

            # Basic example
            results["google_ai_basic"] = google_ai_basic_example()

            # Enhanced example
            results["google_ai_enhanced"] = google_ai_enhanced_example()

        except ImportError:
            print(" Google AI SDK not available")
            print('   Install with: pip install "revenium-python-sdk[google-genai]"')
            results["google_ai_basic"] = False
            results["google_ai_enhanced"] = False

    elif provider == "vertex-ai":
        # Vertex AI SDK Examples
        print("  VERTEX AI SDK EXAMPLES")
        print("=" * 70)

        try:
            import vertexai

            print(" Vertex AI SDK detected")

            # Basic example
            results["vertex_ai_basic"] = vertex_ai_basic_example()

            # Enhanced example
            results["vertex_ai_enhanced"] = vertex_ai_enhanced_example()

        except ImportError:
            print(" Vertex AI SDK not available")
            print('   Install with: pip install "revenium-python-sdk[google-vertex]"')
            results["vertex_ai_basic"] = False
            results["vertex_ai_enhanced"] = False

    # Summary
    print("\n" + "=" * 70)
    print(" TEST RESULTS SUMMARY")
    print("=" * 70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed > 0:
        print(" Success! Check your Revenium dashboard for usage data")

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
        print(" All tests failed. Check your configuration and API keys")
        return 1


if __name__ == "__main__":
    sys.exit(main())
