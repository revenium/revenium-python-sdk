#!/usr/bin/env python3
"""
E2E Test 2.2: Flat Metadata Keys - CURRENT STATE

This tests what the CURRENT examples show (flat keys).
This is the WRONG pattern but we're testing if backward compatibility works.
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


def test_flat_metadata_backward_compatibility():
    """Verify flat keys still work (converted to nested by middleware)."""
    print("=" * 80)
    print("E2E Test 2.2: Flat Metadata Keys - Backward Compatibility")
    print("=" * 80)
    print()
    print("⚠️  NOTE: This tests the CURRENT (WRONG) pattern in examples")
    print("   We're verifying middleware converts flat→nested for backward compat")
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

    print("📝 Making API call with FLAT keys (current example pattern)...")
    print("   Using WRONG pattern: subscriber_id, subscriber_email")
    print("   Middleware should convert these to nested structure")
    print()

    try:
        # Use OLD flat structure (what current examples show)
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": "Test"}],
            usage_metadata={
                "subscriber_id": "test-user-456",
                "subscriber_email": "oldstyle@example.com",
                "subscriber_credential_name": "legacy-api-key",
                "organization_id": "e2e-test-flat-keys",
                "trace_id": "test-flat-keys-001"
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
        print("2. Filter by organization_id: 'e2e-test-flat-keys'")
        print("3. Find event with trace_id: 'test-flat-keys-001'")
        print("4. Check if subscriber fields ARE populated (middleware conversion):")
        print()
        print("   CRITICAL - Did Middleware Convert Flat→Nested?")
        print("   ✓ subscriber.id: 'test-user-456' (from flat subscriber_id)")
        print("   ✓ subscriber.email: 'oldstyle@example.com' (from flat subscriber_email)")
        print("   ✓ subscriber.credential.name: 'legacy-api-key' (from flat)")
        print()
        print("   IF THESE ARE EMPTY:")
        print("   ❌ Middleware conversion NOT working - users lose data!")
        print()
        print("   IF THESE ARE POPULATED:")
        print("   ✅ Backward compatibility working")
        print("   ⚠️  But examples still teach wrong pattern!")
        print()
        print("=" * 80)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_flat_metadata_backward_compatibility()
    sys.exit(0 if success else 1)
