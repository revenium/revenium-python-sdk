"""
 Revenium Google AI Middleware - Embeddings Test Examples

This script demonstrates embeddings functionality with the Revenium middleware for both
Google AI SDK (Gemini Developer API) and Vertex AI SDK.

Requirements:
- REVENIUM_METERING_API_KEY environment variable
- For Google AI: GOOGLE_API_KEY
- For Vertex AI: GOOGLE_CLOUD_PROJECT and authentication

Usage:
    python simple_embeddings_test.py                    # Test Google AI SDK (default)
    python simple_embeddings_test.py --provider google-ai    # Test Google AI SDK explicitly
    python simple_embeddings_test.py --provider vertex-ai    # Test Vertex AI SDK
    python simple_embeddings_test.py --help                  # Show help
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
        description=" Revenium Google AI Middleware Embeddings Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Test Google AI SDK embeddings (default)
  %(prog)s --provider google-ai      # Test Google AI SDK embeddings explicitly
  %(prog)s --provider vertex-ai      # Test Vertex AI SDK embeddings

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
   PASS: Successful embeddings API calls with token counting and Revenium metering
   FAIL: Missing environment variables or authentication issues

Note: Vertex AI SDK provides superior token counting for embeddings operations!

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


def simple_embeddings_test():
    """Simple embeddings test with Google AI"""
    print("\n Google AI SDK - Embeddings Test")
    print("=" * 50)

    google_api_key = os.getenv("GOOGLE_API_KEY")

    try:
        # Import the middleware (this activates it)
        print("\n Importing Revenium middleware...")
        import revenium_middleware.google

        print(" Middleware imported and activated")

        # Import Google AI SDK
        print(" Importing Google AI SDK...")
        from google import genai

        print(" Google AI SDK imported")

        # Create client
        print("Creating Google AI client...")
        client = genai.Client(api_key=google_api_key)
        print(" Gemini Developer API client created")

        # Make embeddings API call with Revenium metadata
        print(" Making embeddings API call...")
        test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for embeddings."

        response = client.models.embed_content(
            model="text-embedding-004",
            contents=test_text,
            usage_metadata={
                "trace_id": "embeddings-test-001",
                "task_type": "document-embedding",
                "subscriber": {
                    "email": "test@example.com",
                },
                "organization_name": "test-org",
                "agent": "embeddings-test-bot",
            },
        )

        # Access the first embedding from the embeddings list
        if response.embeddings and len(response.embeddings) > 0:
            embedding_values = response.embeddings[0].values
            print(f" Generated embedding with {len(embedding_values)} dimensions")
            print(f" Input text: '{test_text[:50]}...'")
            print(f" First few embedding values: {embedding_values[:5]}")
        else:
            print("  No embeddings found in response")

        # Show token usage - Google AI SDK limitation for embeddings
        print("  Token usage: Not available (Google AI SDK limitation)")
        print(
            "   Note: Vertex AI REST API has token counts, but Google AI SDK doesn't expose them"
        )

        print("\n Embeddings test completed successfully!")
        print("   Check your Revenium dashboard for metering data")
        return True

    except Exception as e:
        print(f" Embeddings test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def simple_vertex_ai_embeddings_test():
    """Simple embeddings test with Vertex AI SDK - demonstrates full token counting!"""
    print("\n Simple Vertex AI Embeddings Test")
    print("=" * 40)

    # Check required environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

    if not project_id:
        print(" GOOGLE_CLOUD_PROJECT not found in environment")
        return False

    if not revenium_key:
        print("  REVENIUM_METERING_API_KEY not found - metering will fail")

    print(f" Project ID: {project_id}")
    print(f" Location: {location}")
    if revenium_key:
        print(f" Revenium Key: {revenium_key[:8]}...")

    try:
        # Import the middleware (this activates it)
        print("\n Importing Revenium middleware...")
        import revenium_middleware.google

        print(" Middleware imported and activated")

        # Import Vertex AI SDK
        print(" Importing Vertex AI SDK...")
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        print(" Vertex AI SDK imported")

        # Initialize Vertex AI
        print("Initializing Vertex AI...")
        vertexai.init(project=project_id, location=location)
        print(" Vertex AI initialized")

        # Create embedding model
        print(" Creating Vertex AI embedding model...")
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        print(" Embedding model created")

        # Make embeddings API call
        print(" Making embeddings API call...")
        test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for embeddings."

        # Get embeddings
        embeddings = model.get_embeddings([test_text])

        if embeddings and len(embeddings) > 0:
            embedding_values = embeddings[0].values
            print(f" Generated embedding with {len(embedding_values)} dimensions")
            print(f" Input text: '{test_text[:50]}...'")
            print(f" First few embedding values: {embedding_values[:5]}")

            # KEY ADVANTAGE: Vertex AI SDK provides token counts for embeddings!
            if hasattr(embeddings[0], "statistics") and embeddings[0].statistics:
                stats = embeddings[0].statistics
                if hasattr(stats, "token_count"):
                    print(f" TOKEN COUNT AVAILABLE: {stats.token_count} tokens")
                    print(
                        "   This is a key advantage of Vertex AI SDK over Google AI SDK!"
                    )
                else:
                    print("  Token count not available in statistics")
            else:
                print("  No statistics available in embedding response")
        else:
            print("  No embeddings found in response")

        print("\n Vertex AI embeddings test completed successfully!")
        print("   Check your Revenium dashboard for metering data")
        print("   Vertex AI SDK provides full token counting for embeddings!")
        return True

    except Exception as e:
        print(f" Vertex AI embeddings test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run embeddings tests for the selected provider"""
    parser = create_argument_parser()
    args = parser.parse_args()

    provider = args.provider

    print(" Revenium Google AI Middleware - Embeddings Test Suite")
    print(f"Testing: {provider.upper().replace('-', ' ')} SDK Embeddings")
    if provider == "vertex-ai":
        print("Vertex AI SDK provides superior token counting for embeddings!")
    print("=" * 70)

    # Validate environment for selected provider
    if not validate_environment(provider):
        print("\n Environment validation failed")
        print("   Please fix the configuration issues above and try again")
        return 1

    print()  # Add spacing after environment validation

    success = False

    if provider == "google-ai":
        print(" GOOGLE AI SDK EMBEDDINGS TEST")
        print("=" * 70)

        try:
            import google.genai  # noqa: F401 - checking availability

            print(" Google AI SDK detected")
            success = simple_embeddings_test()
        except ImportError:
            print(" Google AI SDK not available")
            print('   Install with: pip install "revenium-python-sdk[google-genai]"')

    elif provider == "vertex-ai":
        print("  VERTEX AI SDK EMBEDDINGS TEST")
        print("=" * 70)

        try:
            import vertexai  # noqa: F401 - checking availability

            print(" Vertex AI SDK detected")
            success = simple_vertex_ai_embeddings_test()
        except ImportError:
            print(" Vertex AI SDK not available")
            print('   Install with: pip install "revenium-python-sdk[google-vertex]"')

    # Summary
    print("\n" + "=" * 70)
    print(" EMBEDDINGS TEST RESULTS")
    print("=" * 70)

    if success:
        print(" PASS: Embeddings test successful!")
        if provider == "vertex-ai":
            print("Vertex AI SDK provides complete token counting for embeddings!")
        print(" Check your Revenium dashboard for embeddings usage data")

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
        print(" FAIL: Embeddings test failed")
        print("   Check your configuration and API keys")
        return 1


if __name__ == "__main__":
    sys.exit(main())
