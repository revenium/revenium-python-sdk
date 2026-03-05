#!/usr/bin/env python3
"""
E2E Test 3.1: Streaming Token Counting

Verify streaming responses are metered correctly.
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


def test_streaming_token_metering():
    """Verify streaming responses are correctly metered."""
    print("=" * 80)
    print("E2E Test 3.1: Streaming Token Counting")
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

    print("📝 Making STREAMING API call...")
    print("   Testing token counting accuracy for streaming")
    print()

    try:
        with client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            usage_metadata={
                "organization_id": "e2e-test-streaming",
                "trace_id": "test-streaming-001",
                "task_type": "streaming-test"
            }
        ) as stream:
            print("   Streaming response: ", end="", flush=True)
            full_text = ""
            for text in stream.text_stream:
                full_text += text
                print(text, end="", flush=True)
            print()

        # Get final message with usage
        final_message = stream.get_final_message()

        assert final_message.usage.input_tokens > 0, "No input tokens"
        assert final_message.usage.output_tokens > 0, "No output tokens"
        assert len(full_text) > 0, "No response text"

        print()
        print("✅ Streaming call successful!")
        print(f"   Full response: {full_text}")
        print(f"   Input tokens: {final_message.usage.input_tokens}")
        print(f"   Output tokens: {final_message.usage.output_tokens}")
        print()

        print("=" * 80)
        print("MANUAL VERIFICATION REQUIRED")
        print("=" * 80)
        print()
        print("1. Go to Revenium dashboard")
        print("2. Filter by organization_id: 'e2e-test-streaming'")
        print("3. Find event with trace_id: 'test-streaming-001'")
        print("4. Verify streaming-specific fields:")
        print()
        print("   CRITICAL - Streaming Fields:")
        print(f"   ✓ isStreamed: true")
        print(f"   ✓ inputTokenCount: {final_message.usage.input_tokens}")
        print(f"   ✓ outputTokenCount: {final_message.usage.output_tokens}")
        print(f"   ✓ timeToFirstToken: > 0 (should be populated)")
        print()
        print("5. Verify token counts match Anthropic response above")
        print()
        print("=" * 80)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_streaming_token_metering()
    sys.exit(0 if success else 1)
