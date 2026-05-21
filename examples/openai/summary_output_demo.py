#!/usr/bin/env python3
"""
Terminal Summary Output Demo

Demonstrates the terminal summary feature that prints cost/metrics after each
API request. Supports both human-readable and JSON output formats.

Configuration:
- REVENIUM_PRINT_SUMMARY: 'human'/'true' for readable, 'json' for JSON,
  'false' to disable
- REVENIUM_TEAM_ID: Required to fetch and display cost information
"""
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.openai.middleware


def demo_human_format():
    """Demonstrate human-readable summary format."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Human-Readable Format")
    print("="*70)
    print("Setting REVENIUM_PRINT_SUMMARY=human\n")

    os.environ['REVENIUM_PRINT_SUMMARY'] = 'human'

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ],
        max_tokens=10
    )

    print(f"\nAPI Response: {response.choices[0].message.content}")
    print("(Summary appears above)")

    # Wait for async summary to complete
    time.sleep(2)


def demo_json_format():
    """Demonstrate JSON summary format."""
    print("\n" + "="*70)
    print("EXAMPLE 2: JSON Format")
    print("="*70)
    print("Setting REVENIUM_PRINT_SUMMARY=json\n")

    os.environ['REVENIUM_PRINT_SUMMARY'] = 'json'

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": "What is 3+3? Answer in one word."}
        ],
        max_tokens=10
    )

    print(f"\nAPI Response: {response.choices[0].message.content}")
    print("(JSON summary appears above)")

    # Wait for async summary to complete
    time.sleep(2)


def demo_disabled():
    """Demonstrate summary disabled."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Summary Disabled")
    print("="*70)
    print("Setting REVENIUM_PRINT_SUMMARY=false\n")

    os.environ['REVENIUM_PRINT_SUMMARY'] = 'false'

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": "What is 5+5? Answer in one word."}
        ],
        max_tokens=10
    )

    print(f"\nAPI Response: {response.choices[0].message.content}")
    print("(No summary should appear)")

    # Wait for async operations to complete
    time.sleep(2)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("REVENIUM TERMINAL SUMMARY OUTPUT - DEMO")
    print("="*70)

    demo_human_format()
    demo_json_format()
    demo_disabled()

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nNotes:")
    print("- Set REVENIUM_TEAM_ID in .env to see cost information")
    print("- Find your team ID in the Revenium web app")
    print("- Cost data may show as 'Pending' while aggregating\n")


if __name__ == "__main__":
    main()

