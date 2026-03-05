#!/usr/bin/env python3
"""
Summary Output Demo

This example demonstrates the terminal summary output feature that displays
cost and metrics information after each API request.

Supported formats:
- Human-readable: Professional text output with clear formatting
- JSON: Machine-readable single-line JSON output
- Disabled: No output (default)

Configuration:
- REVENIUM_PRINT_SUMMARY: Controls output format
  - 'human' or 'true' → Human-readable format
  - 'json' → JSON format
  - 'false' or unset → Disabled (default)
- REVENIUM_TEAM_ID: Optional, required to fetch and display cost information

Usage:
    # Human-readable format
    REVENIUM_PRINT_SUMMARY=human python examples/summary_output_demo.py

    # JSON format
    REVENIUM_PRINT_SUMMARY=json python examples/summary_output_demo.py

    # Disabled (default)
    REVENIUM_PRINT_SUMMARY=false python examples/summary_output_demo.py
"""

import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables.")

# Import the middleware BEFORE importing Google AI SDK
import revenium_middleware.google  # noqa: F401

# Check which SDK is available
try:
    from google import genai
    SDK_AVAILABLE = "google_ai"
except ImportError:
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        SDK_AVAILABLE = "vertex_ai"
    except ImportError:
        print("Error: Neither Google AI SDK nor Vertex AI SDK is installed.")
        print("Install with: pip install 'revenium-python-sdk[google-genai]'")
        print("         or: pip install 'revenium-python-sdk[google-vertex]'")
        sys.exit(1)


def demo_google_ai():
    """Demo using Google AI SDK."""
    print("\n" + "=" * 60)
    print("Google AI SDK Demo")
    print("=" * 60)

    client = genai.Client()

    # Make a simple request
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="What is 2 + 2? Answer in one word.",
    )

    print(f"\nResponse: {response.text}")


def demo_vertex_ai():
    """Demo using Vertex AI SDK."""
    print("\n" + "=" * 60)
    print("Vertex AI SDK Demo")
    print("=" * 60)

    # Initialize Vertex AI
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable not set.")
        return

    vertexai.init(project=project, location=location)

    # Create model and make request
    model = GenerativeModel("gemini-2.0-flash-001")
    response = model.generate_content("What is 2 + 2? Answer in one word.")

    print(f"\nResponse: {response.text}")


def main():
    """Main demo function."""
    # Show current configuration
    print_summary = os.getenv("REVENIUM_PRINT_SUMMARY", "false")
    team_id = os.getenv("REVENIUM_TEAM_ID", "not set")

    print("\n" + "=" * 60)
    print("TERMINAL SUMMARY OUTPUT DEMO")
    print("=" * 60)
    print(f"\nCurrent Configuration:")
    print(f"  REVENIUM_PRINT_SUMMARY: {print_summary}")
    print(f"  REVENIUM_TEAM_ID: {team_id}")
    print(f"  SDK Available: {SDK_AVAILABLE}")

    if print_summary.lower() in ("false", "0", "no", "off", ""):
        print("\n⚠️  Summary output is DISABLED.")
        print("   Set REVENIUM_PRINT_SUMMARY=human or REVENIUM_PRINT_SUMMARY=json to enable.")
    elif print_summary.lower() in ("true", "1", "yes", "on", "human"):
        print("\n✓ Summary output is ENABLED (human-readable format)")
    elif print_summary.lower() == "json":
        print("\n✓ Summary output is ENABLED (JSON format)")

    if team_id == "not set":
        print("\n⚠️  REVENIUM_TEAM_ID is not set.")
        print("   Cost information will not be available.")

    # Run the appropriate demo
    if SDK_AVAILABLE == "google_ai":
        demo_google_ai()
    else:
        demo_vertex_ai()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

