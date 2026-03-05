"""
Streaming Request Example

This example demonstrates how the middleware works with streaming responses.
The middleware automatically tracks the entire stream as a single request
in Revenium.

Run this example:
    python example_streaming.py

Expected output:
    - Streaming chat response (chunks printed as received)
    - Automatic usage tracking of the entire stream
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
print("STREAMING REQUEST EXAMPLE")
print("=" * 80)
print()

# Make a streaming chat request
question = "Count to 5."
print(f"Question: {question}")
print("Answer (streaming): ", end="", flush=True)

stream = ollama.chat(
    model='qwen2.5:0.5b',
    messages=[
        {
            'role': 'user',
            'content': question,
        },
    ],
    stream=True
)

# Process the stream
chunk_count = 0

for chunk in stream:
    # Print the chunk content
    if 'message' in chunk and 'content' in chunk['message']:
        print(chunk['message']['content'], end="", flush=True)

    chunk_count += 1

print()
print()

print(f"Total chunks received: {chunk_count}")
print()

print("In Revenium, this stream will be recorded as:")
print("  - Single transaction")
print("  - Total tokens for the entire stream")
print("  - Complete request/response metadata")
print()
print("=" * 80)
