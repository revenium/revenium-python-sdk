#!/usr/bin/env python3
"""
Test dual SDK support for Revenium Perplexity middleware.

This test verifies that both OpenAI SDK and native Perplexity SDK
wrappers are correctly applied and functional.
"""
import os
import sys
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the middleware - this should apply both wrappers
import revenium_middleware.perplexity


@pytest.mark.e2e
def test_openai_sdk():
    """Test OpenAI SDK wrapper."""
    print("\n" + "="*60)
    print("Testing OpenAI SDK Wrapper")
    print("="*60)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )

        print("✓ OpenAI client created")

        # Make a simple request
        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "user", "content": "Say 'Hello from OpenAI SDK' in one word"}
            ]
        )

        print(f"✓ Request successful")
        print(f"  Response: {response.choices[0].message.content}")
        print(f"  Tokens: {response.usage.total_tokens}")
        print(f"  Model: {response.model}")

        # Use assertions instead of returning True
        assert response is not None
        assert response.choices[0].message.content is not None

    except ImportError as e:
        print(f"✗ OpenAI SDK not installed: {e}")
        print("  Install with: pip install openai")
        raise
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


@pytest.mark.e2e
def test_perplexity_sdk():
    """Test native Perplexity SDK wrapper."""
    print("\n" + "="*60)
    print("Testing Native Perplexity SDK Wrapper")
    print("="*60)

    try:
        from perplexity import Perplexity

        client = Perplexity(
            api_key=os.getenv("PERPLEXITY_API_KEY")
        )

        print("✓ Perplexity client created")

        # Make a simple request
        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "user", "content": "Say 'Hello from Perplexity SDK' in one word"}
            ]
        )

        print(f"✓ Request successful")
        print(f"  Response: {response.choices[0].message.content}")
        print(f"  Tokens: {response.usage.total_tokens}")
        print(f"  Model: {response.model}")

        # Use assertions instead of returning True
        assert response is not None
        assert response.choices[0].message.content is not None

    except ImportError as e:
        print(f"✗ Perplexity SDK not installed: {e}")
        print("  Install with: pip install perplexityai")
        raise
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Dual SDK Support Test Suite")
    print("="*60)

    # Check environment
    if not os.getenv("PERPLEXITY_API_KEY"):
        print("\n✗ PERPLEXITY_API_KEY not set in environment")
        print("  Please set it in your .env file")
        sys.exit(1)

    if not os.getenv("REVENIUM_METERING_API_KEY"):
        print("\n✗ REVENIUM_METERING_API_KEY not set in environment")
        print("  Please set it in your .env file")
        sys.exit(1)

    print("✓ Environment variables configured")

    # Run tests
    results = {}

    try:
        test_openai_sdk()
        results["OpenAI SDK"] = True
    except Exception:
        results["OpenAI SDK"] = False

    try:
        test_perplexity_sdk()
        results["Perplexity SDK"] = True
    except Exception:
        results["Perplexity SDK"] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ All tests passed!")
        print("Both SDK wrappers are working correctly.")
    else:
        print("\n⚠️  Some tests failed.")
        print("Install missing SDKs or check error messages above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

