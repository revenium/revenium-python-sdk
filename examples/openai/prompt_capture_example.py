"""
Example demonstrating prompt capture functionality.

This example shows how to enable prompt capture to store prompts and responses
in Revenium for analytics and debugging purposes.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import OpenAI middleware FIRST to register wrappers
import revenium_middleware.openai
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("=" * 70)
print("OpenAI Middleware - Prompt Capture Example")
print("=" * 70)
print()

# Example 1: Prompt capture disabled (default)
print("=== Example 1: Prompt Capture Disabled (Default) ===")
print("Environment: REVENIUM_CAPTURE_PROMPTS not set or set to 'false'")
print()

response = client.chat.completions.create(
    model="gpt-5.5",
    max_tokens=100,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    usage_metadata={
        "organizationName": "DemoOrg",
        "productName": "prompt-capture-demo"
    }
)

print(f"Response: {response.choices[0].message.content}")
print("Note: Prompts and responses are NOT captured in Revenium")
print()

# Example 2: Prompt capture enabled
print("=== Example 2: Prompt Capture Enabled ===")
print("To enable prompt capture, set environment variable:")
print("  export REVENIUM_CAPTURE_PROMPTS=true")
print()
print("When enabled, the following data is captured:")
print("  - System prompt (from system message)")
print("  - Input messages (user/assistant messages as JSON)")
print("  - Output response (assistant's response)")
print("  - Truncation flag (if any field exceeded 50,000 characters)")
print()

# Check if prompt capture is enabled
capture_enabled = os.getenv("REVENIUM_CAPTURE_PROMPTS", "false").lower() in ("true", "1", "yes", "on")

if capture_enabled:
    print("✅ Prompt capture is ENABLED")
    print()
    
    response = client.chat.completions.create(
        model="gpt-5.5",
        max_tokens=100,
        messages=[
            {"role": "system", "content": "You are a geography expert."},
            {"role": "user", "content": "What is the capital of Spain?"}
        ],
        usage_metadata={
            "organizationName": "DemoOrg",
            "productName": "prompt-capture-demo",
            "trace_id": "capture-enabled-example"
        }
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print()
    print("✅ Prompts and responses are captured in Revenium!")
    print("   Check your Revenium dashboard to see the captured data.")
else:
    print("⚠️  Prompt capture is DISABLED")
    print("   To enable, set: export REVENIUM_CAPTURE_PROMPTS=true")
    print("   Then run this example again.")

print()

# Example 3: Streaming with prompt capture
print("=== Example 3: Streaming with Prompt Capture ===")
print()

if capture_enabled:
    print("Streaming response with prompt capture enabled...")
    print()
    
    stream = client.chat.completions.create(
        model="gpt-5.5",
        max_tokens=100,
        stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        usage_metadata={
            "organizationName": "DemoOrg",
            "productName": "prompt-capture-demo",
            "trace_id": "streaming-capture-example"
        }
    )
    
    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    print()
    print("✅ Streaming prompts and responses are captured in Revenium!")
else:
    print("⚠️  Prompt capture is disabled. Enable it to see streaming capture.")

print()

# Example 4: Truncation behavior
print("=== Example 4: Truncation Behavior ===")
print()
print("Each field (system prompt, input messages, output response) has a")
print("maximum length of 50,000 characters.")
print()
print("If any field exceeds this limit:")
print("  - The field is truncated to 50,000 characters")
print("  - The 'promptsTruncated' flag is set to true")
print("  - This flag is sent to Revenium for visibility")
print()

if capture_enabled:
    # Create a long system prompt
    long_prompt = "You are a helpful assistant. " * 2000  # ~56,000 chars
    
    response = client.chat.completions.create(
        model="gpt-5.5",
        max_tokens=50,
        messages=[
            {"role": "system", "content": long_prompt},
            {"role": "user", "content": "Say hello."}
        ],
        usage_metadata={
            "organizationName": "DemoOrg",
            "productName": "prompt-capture-demo",
            "trace_id": "truncation-example"
        }
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print()
    print(f"System prompt length: {len(long_prompt)} characters")
    print("✅ System prompt was truncated to 50,000 characters")
    print("✅ promptsTruncated flag set to true in Revenium")
else:
    print("⚠️  Enable prompt capture to see truncation behavior")

print()
print("=" * 70)
print("Example complete!")
print("=" * 70)

