"""
Example demonstrating tool metering with the Anthropic middleware.

This example shows how to use the @meter_tool decorator to meter arbitrary
tool/function calls (database lookups, API calls, etc.) alongside your
automatic LLM API metering.

Run this example:
    python example_meter_tool.py

Prerequisites:
    export ANTHROPIC_API_KEY="your-anthropic-api-key"
    export REVENIUM_METERING_API_KEY="your-revenium-api-key"
    export REVENIUM_METERING_BASE_URL="https://api.revenium.io/meter"
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Anthropic middleware FIRST to register wrappers
import revenium_middleware.anthropic
from anthropic import Anthropic

# Import tool metering utilities
from revenium_middleware import meter_tool, configure

# Configure the metering client for tool calls
api_key = os.getenv("REVENIUM_METERING_API_KEY", "demo-key")
if api_key == "demo-key":
    print("\n⚠️  WARNING: Using demo API key. Set REVENIUM_METERING_API_KEY for real metering.\n")

configure(
    metering_url=os.getenv("REVENIUM_METERING_BASE_URL", "https://api.revenium.io/meter"),
    api_key=api_key,
)

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@meter_tool("customer-database", operation="lookup", agent="support-bot")
def lookup_customer(customer_id: str) -> dict:
    """
    Simulated database lookup tool.

    The @meter_tool decorator automatically captures:
    - Execution time (duration_ms)
    - Success/failure status
    - Attribution metadata (agent, organization, etc.)
    """
    print(f"  Looking up customer: {customer_id}")
    # Simulate database query
    time.sleep(0.3)
    return {
        "customer_id": customer_id,
        "name": "Jane Smith",
        "plan": "Enterprise",
        "active_since": "2024-03-15",
        "recent_tickets": 3,
    }


def analyze_with_claude(customer_data: dict, question: str) -> str:
    """
    Use Claude to analyze customer data.

    This call is automatically metered by the Anthropic middleware.
    """
    context = (
        f"Customer: {customer_data['name']}\n"
        f"Plan: {customer_data['plan']}\n"
        f"Active since: {customer_data['active_since']}\n"
        f"Recent tickets: {customer_data['recent_tickets']}"
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Given this customer data:\n{context}\n\n"
                    f"Answer briefly: {question}"
                ),
            }
        ],
    )
    return response.content[0].text


def main():
    """Run the tool metering example."""
    print("=" * 60)
    print("Anthropic Middleware - Tool Metering Example")
    print("=" * 60)

    # Step 1: Look up customer in database (metered via @meter_tool)
    print("\nStep 1: Database lookup (metered as tool call)...")
    customer = lookup_customer("CUST-42")
    print(f"  Found: {customer['name']} ({customer['plan']} plan)")

    # Step 2: Analyze with Claude (metered via Anthropic middleware)
    print("\nStep 2: Analyzing with Claude (metered as LLM call)...")
    analysis = analyze_with_claude(
        customer, "Should we offer this customer a discount?"
    )
    print(f"  Analysis: {analysis}")

    print("\n" + "=" * 60)
    print("Both tool calls and LLM calls are metered in Revenium!")
    print("  - Database lookup: tracked via @meter_tool")
    print("  - Claude analysis: tracked via Anthropic middleware")
    print("Check your Revenium dashboard to see the usage data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
