#!/usr/bin/env python3
"""
E2E Test 1.1: Required Fields Validation

Verify all 13 required fields are sent correctly to Revenium API.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_env():
    """Load environment variables from .env file."""
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value


def test_required_fields_are_sent():
    """Verify all required fields are present in metering request."""
    print("=" * 80)
    print("E2E Test 1.1: Required Fields Validation")
    print("=" * 80)
    print()

    # Load environment
    load_env()

    # Verify API keys present
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        return False
    if not os.getenv("REVENIUM_METERING_API_KEY"):
        print("❌ REVENIUM_METERING_API_KEY not set")
        return False

    print("✅ API keys found")
    print()

    # Enable debug logging to capture API requests
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')

    # Import after setting up logging
    import anthropic
    import revenium_middleware.anthropic

    # Check middleware initialization
    if revenium_middleware.anthropic.is_initialized():
        print("✅ Revenium middleware initialized")
    else:
        print("⚠️  Revenium middleware not fully initialized")
    print()

    client = anthropic.Anthropic()

    print("📝 Making API call to Anthropic...")
    print("   Model: claude-3-5-sonnet-20241022")
    print("   Max tokens: 50")
    print("   Metadata: organization_id, trace_id")
    print()

    try:
        # Make a simple request
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello"}],
            usage_metadata={
                "organization_id": "e2e-test-required-fields",
                "trace_id": "test-required-fields-001"
            }
        )

        # Verify response
        assert message.content[0].text, "No response text"
        assert message.usage.input_tokens > 0, "No input tokens"
        assert message.usage.output_tokens > 0, "No output tokens"

        print("✅ API call successful!")
        print(f"   Response: {message.content[0].text[:100]}...")
        print(f"   Input tokens: {message.usage.input_tokens}")
        print(f"   Output tokens: {message.usage.output_tokens}")
        print(f"   Total tokens: {message.usage.input_tokens + message.usage.output_tokens}")
        print()

        print("=" * 80)
        print("MANUAL VERIFICATION REQUIRED")
        print("=" * 80)
        print()
        print("1. Go to Revenium dashboard")
        print("2. Filter by organization_id: 'e2e-test-required-fields'")
        print("3. Find event with trace_id: 'test-required-fields-001'")
        print("4. Verify all required fields are populated:")
        print()
        print("   Required Fields (13):")
        print("   ✓ model - Should be 'claude-3-haiku-20240307'")
        print("   ✓ provider - Should be 'ANTHROPIC'")
        print("   ✓ inputTokenCount - Should be", message.usage.input_tokens)
        print("   ✓ outputTokenCount - Should be", message.usage.output_tokens)
        print("   ✓ totalTokenCount - Should be", message.usage.input_tokens + message.usage.output_tokens)
        print("   ✓ stopReason - Should be 'END' or similar")
        print("   ✓ requestTime - ISO 8601 timestamp")
        print("   ✓ completionStartTime - ISO 8601 timestamp")
        print("   ✓ responseTime - ISO 8601 timestamp")
        print("   ✓ requestDuration - Milliseconds (> 0)")
        print("   ✓ isStreamed - Should be false")
        print("   ✓ costType - Should be 'AI'")
        print("   ✓ transactionId - UUID format")
        print()
        print("5. Event should appear within 30 seconds")
        print()
        print("=" * 80)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_required_fields_are_sent()
    sys.exit(0 if success else 1)
