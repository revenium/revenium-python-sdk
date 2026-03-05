"""
Enhanced Tracking with Metadata

This example demonstrates how to use metadata for advanced tracking and reporting.
Metadata allows you to add custom information to requests for better analytics
in Revenium.

Run this example:
    python example_metadata.py

Expected output:
    - A chat response
    - Confirmation that metadata was sent to Revenium
    - Examples of what you can do with metadata
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
print("REQUEST WITH METADATA")
print("=" * 80)
print()

# Make a chat request with metadata
question = "Explain Python in 5 words."
print(f"Question: {question}")

response = ollama.chat(
    model='qwen2.5:0.5b',
    messages=[
        {
            'role': 'user',
            'content': question,
        },
    ],
    usage_metadata={
        "subscriber": {
            "id": "user-12345",
            "email": "user@example.com"
        },
        "trace_id": "trace-abc-123",
        "task_type": "explanation",
        "organizationName": "AcmeCorp",
        "productName": "demo-chatbot"
    }
)

# Extract and display the response
answer = response['message']['content']
print(f"Answer: {answer}")
print()

print("Metadata sent with this request:")
print("  - Subscriber ID: user-12345")
print("  - Subscriber Email: user@example.com")
print("  - Trace ID: trace-abc-123")
print("  - Task Type: explanation")
print("  - Organization Name: AcmeCorp")
print("  - Product Name: demo-chatbot")
print()

print("In Revenium, you can now:")
print("  - Filter requests by subscriber, organization, or task type")
print("  - Correlate with your distributed tracing system using trace_id")
print("  - Analyze usage patterns by task type")
print("  - Track costs per subscriber or organization")
print()
print("=" * 80)
