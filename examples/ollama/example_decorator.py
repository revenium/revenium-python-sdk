"""
Decorator Example - NEW FEATURE!

This example demonstrates the new decorator functionality for automatic metadata injection.
With decorators, you don't need to pass metadata to every API call - it's automatically
injected for all calls within the decorated function.

Run this example:
    python example_decorator.py

Expected output:
    - Multiple chat responses
    - All automatically tagged with the same metadata
    - Confirmation that metadata was sent to Revenium
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import ollama
import revenium_middleware.ollama
from revenium_middleware import revenium_metadata

# Ensure REVENIUM_METERING_API_KEY is set in your .env file
if not os.getenv("REVENIUM_METERING_API_KEY"):
    raise ValueError(
        "REVENIUM_METERING_API_KEY environment variable is not set. "
        "Please set it in your .env file."
    )

print("=" * 80)
print("DECORATOR EXAMPLE - AUTOMATIC METADATA INJECTION")
print("=" * 80)
print()

# Example 1: Single function with metadata
@revenium_metadata(
    trace_id="session-abc-123",
    task_type="customer-support",
    organization_name="AcmeCorp",
    agent="support-bot-v2",
    product_name="customer-support-chatbot"
)
def handle_customer_query(question):
    """
    All ollama.chat() calls inside this function will automatically
    include the metadata defined in the decorator.
    """
    print(f"Customer Question: {question}")
    
    response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[{'role': 'user', 'content': question}]
    )
    
    answer = response.message.content
    print(f"Bot Answer: {answer}")
    print(f"Transaction ID: {response._revenium_transaction_id}")
    print()
    
    return answer

# Example 2: Multi-step conversation with metadata
@revenium_metadata(
    trace_id="conversation-xyz-789",
    task_type="multi-turn-chat",
    organization_name="DemoOrg",
    subscription_id="sub-premium-001"
)
def multi_turn_conversation():
    """
    All API calls in this function share the same metadata.
    Perfect for tracking entire conversations!
    """
    print("Starting multi-turn conversation...")
    print()
    
    # First turn
    response1 = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[{'role': 'user', 'content': 'What is Python?'}]
    )
    print(f"Turn 1: {response1.message.content[:100]}...")
    
    # Second turn
    response2 = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[{'role': 'user', 'content': 'What is it used for?'}]
    )
    print(f"Turn 2: {response2.message.content[:100]}...")
    
    print()
    print("✅ Both turns automatically tagged with the same metadata!")
    print()

# Run examples
print("Example 1: Single Query with Metadata")
print("-" * 80)
handle_customer_query("How do I reset my password?")

print("Example 2: Multi-Turn Conversation")
print("-" * 80)
multi_turn_conversation()

print("=" * 80)
print("BENEFITS OF DECORATORS:")
print("=" * 80)
print("✅ No need to pass metadata to every API call")
print("✅ Cleaner code - metadata defined once at function level")
print("✅ Automatic tracking of entire workflows/conversations")
print("✅ Easy to trace multi-step processes in Revenium")
print("✅ Works with both sync and async functions")
print()
print("All metering data with metadata has been sent to Revenium!")
print("=" * 80)

# Give time for background metering calls to complete
import time
time.sleep(2)

