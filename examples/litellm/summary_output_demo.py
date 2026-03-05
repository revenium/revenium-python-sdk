#!/usr/bin/env python
"""
Revenium LiteLLM Middleware - Terminal Summary Output Demo

This example demonstrates the terminal summary output feature that displays
cost and usage metrics after each API request.

Supported Output Formats:
1. Human-readable - Professional formatted text output
2. JSON - Machine-readable single-line JSON output
3. Disabled - No output (default)

Configuration via environment variables:
- REVENIUM_PRINT_SUMMARY: 'human', 'json', 'true', or 'false'
- REVENIUM_TEAM_ID: Required to fetch and display cost information

Usage:
    # Test human-readable format
    REVENIUM_PRINT_SUMMARY=human python examples/summary_output_demo.py

    # Test JSON format
    REVENIUM_PRINT_SUMMARY=json python examples/summary_output_demo.py

    # Test disabled (default)
    REVENIUM_PRINT_SUMMARY=false python examples/summary_output_demo.py
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Import the middleware BEFORE litellm - this enables automatic tracking
import revenium_middleware.litellm.client.middleware  # noqa: F401, E402

# Now import litellm
import litellm  # noqa: E402

# Check required environment variables
proxy_url = os.getenv("LITELLM_PROXY_URL")
proxy_key = os.getenv("LITELLM_API_KEY")

if not proxy_url or not proxy_key:
    print("Error: LITELLM_PROXY_URL and LITELLM_API_KEY must be set in .env file")
    print("See .env.example for configuration details")
    exit(1)

# Configure LiteLLM to use the proxy
litellm.api_base = proxy_url
litellm.api_key = proxy_key

# Display current configuration
print("=" * 60)
print("TERMINAL SUMMARY OUTPUT DEMO")
print("=" * 60)
print(f"REVENIUM_PRINT_SUMMARY: {os.getenv('REVENIUM_PRINT_SUMMARY', 'false (default)')}")
print(f"REVENIUM_TEAM_ID: {'***configured***' if os.getenv('REVENIUM_TEAM_ID') else 'not set'}")
print("=" * 60)
print()


def demo_completion():
    """Run a simple completion to demonstrate summary output."""
    print("Running LiteLLM completion...")
    print("-" * 40)

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'Hello, World!' in exactly 3 words."}
        ],
        usage_metadata={
            "trace_id": "summary-demo-trace-001",
            "task_type": "demo",
            "agent": "summary-demo-agent",
        }
    )

    print("-" * 40)
    print(f"Response: {response.choices[0].message.content}")
    print()

    # Show token usage from response
    print("Token usage from response object:")
    print(f"  Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  Completion tokens: {response.usage.completion_tokens}")
    print(f"  Total tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    print()
    print("Note: The terminal summary (if enabled) will appear after the completion.")
    print()

    demo_completion()

    print()
    print("=" * 60)
    print("Demo complete!")
    print()
    print("To change output format, set REVENIUM_PRINT_SUMMARY environment variable:")
    print("  - 'human' or 'true': Human-readable formatted output")
    print("  - 'json': Machine-readable JSON output")
    print("  - 'false': Disabled (default)")
    print()
    print("To see cost information, set REVENIUM_TEAM_ID to your team ID.")
    print("=" * 60)

