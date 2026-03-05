"""
Example demonstrating tool metering with the Google AI middleware.

This example shows how to use the @meter_tool decorator to meter arbitrary
tool/function calls (document parsers, data processors, etc.) alongside your
automatic LLM API metering.

Prerequisites:
    export GOOGLE_API_KEY="your-google-api-key"
    export REVENIUM_METERING_API_KEY="your-revenium-api-key"
    export REVENIUM_METERING_BASE_URL="https://api.revenium.ai"

Installation:
    pip install revenium-python-sdk[google-genai]

Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

import os
import time

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using environment variables

import revenium_middleware.google
from google import genai

# Import tool metering utilities
from revenium_middleware import meter_tool, configure

# Configure the metering client for tool calls
configure(
    metering_url=os.getenv("REVENIUM_METERING_BASE_URL", "https://api.revenium.ai"),
    api_key=os.environ["REVENIUM_METERING_API_KEY"],  # Fail fast if not set
)

# Create Google AI client
client = genai.Client()


@meter_tool("document-parser", operation="parse", agent="doc-assistant")
def parse_document(file_path: str) -> dict:
    """
    Simulated document parser tool.

    The @meter_tool decorator automatically captures:
    - Execution time (duration_ms)
    - Success/failure status
    - Attribution metadata (agent, organization, etc.)
    """
    print(f"  Parsing document: {file_path}")
    # Simulate document parsing work
    time.sleep(0.4)
    return {
        "file_path": file_path,
        "pages": 12,
        "content": (
            "Q3 2025 Financial Report Summary: Revenue increased 15% year-over-year "
            "to $4.2M. Operating expenses remained flat at $2.8M. Net income grew "
            "to $1.4M, up from $1.1M in Q3 2024. The company expanded into two new "
            "markets and onboarded 150 new enterprise customers."
        ),
        "format": "pdf",
    }


def summarize_with_gemini(text: str) -> str:
    """
    Use Gemini to summarize parsed document content.

    This call is automatically metered by the Google AI middleware.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"Summarize the key points in one sentence:\n\n{text}",
    )
    return response.text


def main():
    """Run the tool metering example."""
    print("=" * 60)
    print("Google AI Middleware - Tool Metering Example")
    print("=" * 60)

    # Step 1: Parse a document (metered via @meter_tool)
    print("\nStep 1: Parsing document (metered as tool call)...")
    parsed = parse_document("reports/q3_2025_financials.pdf")
    print(f"  Parsed: {parsed['pages']} pages ({parsed['format']})")

    # Step 2: Summarize with Gemini (metered via Google AI middleware)
    print("\nStep 2: Summarizing with Gemini (metered as LLM call)...")
    summary = summarize_with_gemini(parsed["content"])
    print(f"  Summary: {summary}")

    print("\n" + "=" * 60)
    print("Both tool calls and LLM calls are metered in Revenium!")
    print("  - Document parser: tracked via @meter_tool")
    print("  - Gemini summary: tracked via Google AI middleware")
    print("Check your Revenium dashboard to see the usage data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
