#!/usr/bin/env python
"""
Revenium LiteLLM Middleware - Getting Started

This example demonstrates automatic metering of LiteLLM API calls.
All requests are tracked and sent to Revenium for analytics.

Usage:
    1. Create a .env file with your API keys (see below)
    2. Run: python getting_started.py

Required .env configuration:
    REVENIUM_METERING_API_KEY=hak_your_revenium_key_here
    REVENIUM_METERING_BASE_URL=https://api.revenium.ai
    LITELLM_PROXY_URL=https://your-litellm-proxy.com
    LITELLM_API_KEY=sk-your-proxy-key
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Import the middleware BEFORE litellm - this enables automatic tracking
import revenium_middleware.litellm.client.middleware

# Now import litellm
import litellm

# Configure LiteLLM to use the proxy
proxy_url = os.getenv("LITELLM_PROXY_URL")
proxy_key = os.getenv("LITELLM_API_KEY")

if not proxy_url or not proxy_key:
    print("Error: LITELLM_PROXY_URL and LITELLM_API_KEY must be set in .env file")
    print("See .env.example for configuration details")
    exit(1)

litellm.api_base = proxy_url
litellm.api_key = proxy_key

# That's it! All litellm.completion calls are now automatically metered.


def main():
    """Simple example demonstrating automatic metering."""

    print("=" * 60)
    print("Revenium LiteLLM Middleware - Getting Started")
    print("=" * 60)
    print(f"Using LiteLLM Proxy: {proxy_url}")

    # Basic completion - automatically metered
    print("\n1. Basic completion (auto-metered):")
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello from Revenium!' in one line."}]
    )
    print(f"   Response: {response.choices[0].message.content}")

    # Completion with metadata - for enhanced analytics
    print("\n2. Completion with metadata (enhanced analytics):")
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        usage_metadata={
            # All fields are optional - use what you need
            "organizationName": "AcmeCorp",
            "subscription_id": "premium-plan",
            "productName": "customer-chatbot",
            "agent": "math-helper",
            "task_type": "calculation",
            "trace_id": "session-001",
            "subscriber": {
                "id": "user-123",
                "email": "user@example.com"
            }
        }
    )
    print(f"   Response: {response.choices[0].message.content}")

    print("\n" + "=" * 60)
    print("Done! Check your Revenium dashboard to see the tracked usage.")
    print("Dashboard: https://app.revenium.ai")
    print("=" * 60)


if __name__ == "__main__":
    main()
