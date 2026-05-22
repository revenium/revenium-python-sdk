#!/usr/bin/env python
"""
Example demonstrating tool metering with the LiteLLM middleware.

This example shows how to use the @meter_tool decorator to meter arbitrary
tool/function calls (API fetchers, data processors, etc.) alongside your
automatic LLM API metering with LiteLLM.

Usage:
    1. Create a .env file with your API keys (see below)
    2. Run: python example_meter_tool.py

Required .env configuration:
    REVENIUM_METERING_API_KEY=hak_your_revenium_key_here
    REVENIUM_METERING_BASE_URL=https://api.revenium.ai
    LITELLM_PROXY_URL=https://your-litellm-proxy.com
    LITELLM_API_KEY=sk-your-proxy-key
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Import the middleware BEFORE litellm - this enables automatic tracking
import revenium_middleware.litellm.client.middleware

# Now import litellm
import litellm

# Import tool metering utilities
from revenium_middleware import meter_tool, configure

# Configure LiteLLM proxy
proxy_url = os.getenv("LITELLM_PROXY_URL")
proxy_key = os.getenv("LITELLM_API_KEY")

if not proxy_url or not proxy_key:
    print("Error: LITELLM_PROXY_URL and LITELLM_API_KEY must be set in .env file")
    sys.exit(1)

litellm.api_base = proxy_url
litellm.api_key = proxy_key

# Configure the metering client for tool calls
configure(
    metering_url=os.getenv("REVENIUM_METERING_BASE_URL", "https://api.revenium.ai"),
    api_key=os.getenv("REVENIUM_METERING_API_KEY", "demo-key"),
)


@meter_tool("rest-api-fetcher", operation="fetch", agent="data-pipeline")
def fetch_from_api(endpoint: str) -> dict:
    """
    Simulated API fetcher tool.

    The @meter_tool decorator automatically captures:
    - Execution time (duration_ms)
    - Success/failure status
    - Attribution metadata (agent, organization, etc.)
    """
    print(f"  Fetching: {endpoint}")
    # Simulate API call
    time.sleep(0.3)
    return {
        "endpoint": endpoint,
        "status": 200,
        "data": {
            "users_active": 1250,
            "requests_today": 45000,
            "avg_response_ms": 120,
            "error_rate": 0.02,
        },
    }


def analyze_with_litellm(data: dict) -> str:
    """
    Use LiteLLM to analyze fetched data.

    This call is automatically metered by the LiteLLM middleware.
    """
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Analyze these API metrics and give a brief health assessment:\n\n"
                    f"Active users: {data['users_active']}\n"
                    f"Requests today: {data['requests_today']}\n"
                    f"Avg response time: {data['avg_response_ms']}ms\n"
                    f"Error rate: {data['error_rate']}"
                ),
            }
        ],
    )
    return response.choices[0].message.content


def main():
    """Run the tool metering example."""
    print("=" * 60)
    print("LiteLLM Middleware - Tool Metering Example")
    print("=" * 60)
    print(f"Using LiteLLM Proxy: {proxy_url}")

    # Step 1: Fetch data from API (metered via @meter_tool)
    print("\nStep 1: Fetching API metrics (metered as tool call)...")
    result = fetch_from_api("https://api.internal.example.com/v1/metrics")
    print(f"  Status: {result['status']} - {len(result['data'])} metrics received")

    # Step 2: Analyze with LiteLLM (metered via LiteLLM middleware)
    print("\nStep 2: Analyzing with LiteLLM (metered as LLM call)...")
    analysis = analyze_with_litellm(result["data"])
    print(f"  Analysis: {analysis}")

    print("\n" + "=" * 60)
    print("Both tool calls and LLM calls are metered in Revenium!")
    print("  - API fetcher: tracked via @meter_tool")
    print("  - LiteLLM analysis: tracked via LiteLLM middleware")
    print("Check your Revenium dashboard to see the usage data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
