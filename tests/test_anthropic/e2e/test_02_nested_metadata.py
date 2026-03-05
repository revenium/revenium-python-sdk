#!/usr/bin/env python3
"""
E2E Test 2.1: Nested Subscriber Metadata Validation

Verify nested subscriber structure is correctly sent to Revenium.
This tests the CORRECT pattern users should use.
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


def test_nested_subscriber_metadata():
    """Verify nested subscriber structure is correctly sent to Revenium."""
    print("=" * 80)
    print("E2E Test 2.1: Nested Subscriber Metadata Validation")
    print("=" * 80)
    print()

    # Load environment
    load_env()

    # Verify API keys
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("REVENIUM_METERING_API_KEY"):
        print("❌ API keys not set")
        return False

    print("✅ API keys found")
    print()

    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    import anthropic
    import revenium_middleware.anthropic

    client = anthropic.Anthropic()

    print("📝 Making API call with NESTED subscriber structure...")
    print("   Using CORRECT pattern: subscriber.id, subscriber.email, subscriber.credential")
    print()

    try:
        # Use CORRECT nested structure
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hello"}],
            usage_metadata={
                "subscriber": {
                    "id": "test-user-123",
                    "email": "testuser@example.com",
                    "credential": {
                        "name": "test-api-key",
                        "value": "test-key-value-123"
                    }
                },
                "organization_id": "e2e-test-nested-subscriber",
                "trace_id": "test-nested-subscriber-001"
            }
        )

        assert message.content[0].text

        print("✅ API call successful!")
        print(f"   Response: {message.content[0].text[:100]}...")
        print(f"   Tokens: {message.usage.input_tokens} + {message.usage.output_tokens}")
        print()

        print("=" * 80)
        print("MANUAL VERIFICATION REQUIRED")
        print("=" * 80)
        print()
        print("1. Go to Revenium dashboard")
        print("2. Filter by organization_id: 'e2e-test-nested-subscriber'")
        print("3. Find event with trace_id: 'test-nested-subscriber-001'")
        print("4. Verify subscriber fields are POPULATED:")
        print()
        print("   CRITICAL - Subscriber Data:")
        print("   ✓ subscriber.id: 'test-user-123'")
        print("   ✓ subscriber.email: 'testuser@example.com'")
        print("   ✓ subscriber.credential.name: 'test-api-key'")
        print("   ✓ subscriber.credential.value: 'test-key-value-123'")
        print()
        print("5. Verify you CAN filter/group by subscriber.id")
        print("6. This proves nested structure is working correctly")
        print()
        print("=" * 80)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_nested_subscriber_metadata()
    sys.exit(0 if success else 1)
