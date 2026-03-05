"""
Example demonstrating tool metering with the Ollama middleware.

This example shows how to use the @meter_tool decorator to meter arbitrary
tool/function calls (file readers, data processors, etc.) alongside your
automatic LLM API metering with Ollama.

Run this example:
    python examples/example_meter_tool.py

Prerequisites:
    - Ollama running locally (default: http://localhost:11434)
    - A model pulled (e.g., ollama pull qwen2.5:0.5b)
    export REVENIUM_METERING_API_KEY="your-revenium-api-key"
    export REVENIUM_METERING_BASE_URL="https://api.revenium.ai"
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import ollama
import revenium_middleware.ollama

# Import tool metering utilities
from revenium_middleware import meter_tool, configure

# Validate required environment variable before configuring
api_key = os.environ.get("REVENIUM_METERING_API_KEY")
if not api_key:
    raise ValueError(
        "REVENIUM_METERING_API_KEY environment variable is not set. "
        "Please set it in your .env file."
    )

# Configure the metering client for tool calls
configure(
    metering_url=os.getenv("REVENIUM_METERING_BASE_URL", "https://api.revenium.ai"),
    api_key=api_key,
)


@meter_tool("file-reader", operation="read", agent="local-assistant")
def read_log_file(file_path: str) -> dict:
    """
    Simulated file reader tool.

    The @meter_tool decorator automatically captures:
    - Execution time (duration_ms)
    - Success/failure status
    - Attribution metadata (agent, organization, etc.)
    """
    print(f"  Reading file: {file_path}")
    # Simulate file reading
    time.sleep(0.2)
    return {
        "file_path": file_path,
        "lines": 47,
        "content": (
            "ERROR 2025-01-15 14:32:01 Connection timeout to database server db-prod-3\n"
            "WARN  2025-01-15 14:32:05 Retrying connection (attempt 2/3)\n"
            "ERROR 2025-01-15 14:32:08 Connection timeout to database server db-prod-3\n"
            "WARN  2025-01-15 14:32:12 Retrying connection (attempt 3/3)\n"
            "ERROR 2025-01-15 14:32:15 All connection attempts failed for db-prod-3"
        ),
        "size_kb": 3.2,
    }


def analyze_with_ollama(log_content: str) -> str:
    """
    Use a local Ollama model to analyze log content.

    This call is automatically metered by the Ollama middleware.
    """
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Analyze these log entries and identify the issue:\n\n{log_content}"
                ),
            }
        ],
    )
    return response['message']['content']


def main():
    """Run the tool metering example."""
    print("=" * 60)
    print("Ollama Middleware - Tool Metering Example")
    print("=" * 60)

    # Step 1: Read a log file (metered via @meter_tool)
    print("\nStep 1: Reading log file (metered as tool call)...")
    log_data = read_log_file("/var/log/app/errors.log")
    print(f"  Read: {log_data['lines']} lines ({log_data['size_kb']} KB)")

    # Step 2: Analyze with local LLM (metered via Ollama middleware)
    print("\nStep 2: Analyzing with Ollama (metered as LLM call)...")
    analysis = analyze_with_ollama(log_data["content"])
    print(f"  Analysis: {analysis}")

    print("\n" + "=" * 60)
    print("Both tool calls and LLM calls are metered in Revenium!")
    print("  - File reader: tracked via @meter_tool")
    print("  - Ollama analysis: tracked via Ollama middleware")
    print("Check your Revenium dashboard to see the usage data.")
    print("=" * 60)

    # Give time for background metering calls to complete
    time.sleep(2)


if __name__ == "__main__":
    main()
