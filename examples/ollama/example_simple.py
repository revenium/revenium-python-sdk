"""
Simple Example

This example demonstrates basic middleware usage with Ollama.
The middleware automatically tracks your Ollama API usage in Revenium.

Run this example:
    python example_simple.py

Expected output:
    - A simple chat response
    - Automatic usage tracking in Revenium
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import ollama
import revenium_middleware.ollama

# Ensure REVENIUM_METERING_API_KEY is set in your .env file
if not os.getenv("REVENIUM_METERING_API_KEY"):
    raise ValueError(
        "REVENIUM_METERING_API_KEY environment variable is not set. "
        "Please set it in your .env file."
    )

print("=" * 80)
print("SIMPLE REQUEST EXAMPLE")
print("=" * 80)
print()

# Make a simple chat request
question = "What is 2+2?"
print(f"Question: {question}")

response = ollama.chat(
    model='qwen2.5:0.5b',
    messages=[
        {
            'role': 'user',
            'content': question,
        },
    ],
)

# Extract and display the response
answer = response['message']['content']
print(f"Answer: {answer}")
print()

# Transaction ID is available in the response if needed for debugging
transaction_id = response._revenium_transaction_id
print(f"Transaction ID: {transaction_id}")
print()

print("Usage is automatically tracked in Revenium dashboard.")
print()
print("=" * 80)
