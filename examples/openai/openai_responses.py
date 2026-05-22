#!/usr/bin/env python3
"""
🚀 OpenAI Responses API Example

Shows how to use Revenium middleware with OpenAI Responses API.
The Responses API is the new recommended API for building agentic applications.
Demonstrates seamless metadata integration - all metadata fields are optional!
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.openai.middleware  # noqa: F401


def openai_responses_example():
    print("🚀 OpenAI Responses API Usage with Seamless Metadata Integration\n")

    # Verify API keys are set
    openai_key = os.getenv("OPENAI_API_KEY")
    revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return

    if not revenium_key:
        print("❌ REVENIUM_METERING_API_KEY not found in environment")
        return

    print(f"✅ OpenAI API Key: {'*' * (len(openai_key) - 4)}{openai_key[-4:]}")
    print(f"✅ Revenium API Key: {'*' * (len(revenium_key) - 4)}{revenium_key[-4:]}")
    print()

    # Create OpenAI client (middleware is automatically active)
    openai_client = OpenAI()

    # Example 1: Basic Responses API call (no metadata)
    print("🚀 Example 1: Basic Responses API call (automatic tracking)")
    print("-" * 70)

    try:
        basic_response = openai_client.responses.create(
            model="gpt-4o-mini",
            input="What is the Responses API in one sentence?"
            # No usage_metadata - still automatically tracked!
        )

        print(f"💬 Response: {basic_response.output_text}")
        print(f"📊 Model: {basic_response.model}")
        print(f"📊 Usage: input={basic_response.usage.input_tokens}, "
              f"output={basic_response.usage.output_tokens}, "
              f"total={basic_response.usage.total_tokens}")
        print("✅ Automatically tracked to Revenium without metadata\n")
    except Exception as e:
        print(f"❌ Error in Example 1: {e}\n")

    # Example 2: Responses API with rich metadata (all optional!)
    print("🚀 Example 2: Responses API with rich metadata")
    print("-" * 70)

    try:
        metadata_response = openai_client.responses.create(
            model="gpt-4o-mini",
            input="Explain the benefits of the Responses API in 2 sentences.",

            # ✨ All metadata fields are optional - add what you need!
            usage_metadata={
                # User tracking (optional)
                "subscriber_id": "user-12345",
                "subscriber_email": "developer@company.com",

                # Business context (optional)
                "organizationName": "my-company",
                "productName": "ai-assistant",

                # Task classification (optional)
                "task_type": "explanation-request",
                "trace_id": f"session-{int(time.time() * 1000)}",

                # Custom fields (optional)
                "agent": "openai-responses-example",
                "response_quality_score": 0.95
            }
        )

        print(f"💬 Response: {metadata_response.output_text}")
        print(f"📊 Model: {metadata_response.model}")
        print(f"📊 Usage: input={metadata_response.usage.input_tokens}, "
              f"output={metadata_response.usage.output_tokens}, "
              f"total={metadata_response.usage.total_tokens}")
        print("✅ Tracked with rich metadata for analytics\n")
    except Exception as e:
        print(f"❌ Error in Example 2: {e}\n")

    # Example 3: Responses API with instructions (cleaner API)
    print("🚀 Example 3: Responses API with instructions (cleaner API)")
    print("-" * 70)

    try:
        instructions_response = openai_client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a helpful assistant that explains technical concepts clearly.",
            input="What is middleware?",

            usage_metadata={
                "organizationName": "my-company",
                "task_type": "technical-explanation",
                "trace_id": f"session-{int(time.time() * 1000)}"
            }
        )

        print(f"💬 Response: {instructions_response.output_text}")
        print(f"📊 Model: {instructions_response.model}")
        print(f"📊 Usage: input={instructions_response.usage.input_tokens}, "
              f"output={instructions_response.usage.output_tokens}, "
              f"total={instructions_response.usage.total_tokens}")
        print("✅ Responses API with instructions tracked successfully\n")
    except Exception as e:
        print(f"❌ Error in Example 3: {e}\n")

    # Example 4: Streaming Responses API
    print("🚀 Example 4: Streaming Responses API")
    print("-" * 70)

    try:
        print("💬 Streaming response: ", end="", flush=True)

        stream_response = openai_client.responses.create(
            model="gpt-4o-mini",
            input="Count from 1 to 5 slowly",
            stream=True,

            usage_metadata={
                "organizationName": "my-company",
                "task_type": "streaming-test",
                "trace_id": f"stream-{int(time.time() * 1000)}"
            }
        )

        # Iterate through the stream
        for chunk in stream_response:
            # Print output text chunks as they arrive
            if hasattr(chunk, 'output') and chunk.output:
                for item in chunk.output:
                    if hasattr(item, 'content') and item.content:
                        for content in item.content:
                            if hasattr(content, 'text'):
                                print(content.text, end="", flush=True)

        print("\n✅ Streaming Responses API tracked successfully\n")
    except Exception as e:
        print(f"❌ Error in Example 4: {e}\n")

    # Example 5: Multi-turn conversation with Responses API
    print("🚀 Example 5: Multi-turn conversation with Responses API")
    print("-" * 70)

    try:
        # First turn
        print("Turn 1: Asking about Python...")
        response1 = openai_client.responses.create(
            model="gpt-4o-mini",
            input="What is Python?",

            usage_metadata={
                "organizationName": "my-company",
                "task_type": "multi-turn-conversation",
                "trace_id": f"conversation-{int(time.time() * 1000)}",
                "agent": "multi-turn-example"
            }
        )

        print(f"💬 Response 1: {response1.output_text[:100]}...")
        print(f"📊 Usage: {response1.usage.total_tokens} tokens\n")

        # Second turn - using previous_response_id for context
        print("Turn 2: Following up on Python...")
        response2 = openai_client.responses.create(
            model="gpt-4o-mini",
            input="What are its main use cases?",
            previous_response_id=response1.id,

            usage_metadata={
                "organizationName": "my-company",
                "task_type": "multi-turn-conversation",
                "trace_id": f"conversation-{int(time.time() * 1000)}",
                "agent": "multi-turn-example"
            }
        )

        print(f"💬 Response 2: {response2.output_text[:100]}...")
        print(f"📊 Usage: {response2.usage.total_tokens} tokens")
        print("✅ Multi-turn conversation tracked successfully\n")
    except Exception as e:
        print(f"❌ Error in Example 5: {e}\n")

    # Summary
    print("🎉 Summary:")
    print("✅ Responses API works with or without metadata")
    print("✅ All metadata fields are optional")
    print("✅ No type casting required - native Python support")
    print("✅ Automatic tracking for all Responses API calls")
    print("✅ Streaming responses fully supported")
    print("✅ Multi-turn conversations with previous_response_id")
    print("✅ Rich business analytics when metadata is provided")


if __name__ == "__main__":
    openai_responses_example()
