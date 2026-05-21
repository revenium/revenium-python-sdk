#!/usr/bin/env python3
"""
Summary Output Demo - Revenium Middleware for Anthropic

This example demonstrates the terminal summary output feature that displays
cost and metrics information after each API request.

Supported output formats:
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
    # Human-readable output
    REVENIUM_PRINT_SUMMARY=human python summary_output_demo.py

    # JSON output
    REVENIUM_PRINT_SUMMARY=json python summary_output_demo.py

    # With cost display (requires team ID)
    REVENIUM_PRINT_SUMMARY=human REVENIUM_TEAM_ID=your-team-id python summary_output_demo.py
"""

import os
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import Anthropic and Revenium middleware
import anthropic
import revenium_middleware.anthropic


def main():
    """Demonstrate the terminal summary output feature."""
    
    # Check configuration
    print_summary = os.getenv("REVENIUM_PRINT_SUMMARY", "false")
    team_id = os.getenv("REVENIUM_TEAM_ID")
    
    print("=" * 60)
    print("SUMMARY OUTPUT DEMO")
    print("=" * 60)
    print(f"REVENIUM_PRINT_SUMMARY: {print_summary}")
    print(f"REVENIUM_TEAM_ID: {'set' if team_id else 'not set'}")
    print("=" * 60)
    print()
    
    if print_summary.lower() in ("false", ""):
        print("Summary output is DISABLED.")
        print("To enable, set REVENIUM_PRINT_SUMMARY to 'human' or 'json'")
        print()
        print("Examples:")
        print("  REVENIUM_PRINT_SUMMARY=human python summary_output_demo.py")
        print("  REVENIUM_PRINT_SUMMARY=json python summary_output_demo.py")
        print()
    
    # Create Anthropic client
    client = anthropic.Anthropic()
    
    print("Making API request...")
    print("-" * 60)
    print()
    
    # Make a simple API call
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
        ],
        usage_metadata={
            "trace_id": "summary-demo-001",
            "organizationName": "demo-org",
            "productName": "summary-demo"
        }
    )
    
    print()
    print("-" * 60)
    print("Response:")
    print(message.content[0].text)
    print()
    
    # If summary is enabled, it will be printed automatically after the API call
    if print_summary.lower() in ("false", ""):
        print("(No summary printed - feature is disabled)")
    else:
        print("(Summary was printed above after the API call)")
    
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

