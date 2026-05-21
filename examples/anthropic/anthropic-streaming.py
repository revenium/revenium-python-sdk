#!/usr/bin/env python3
"""
Anthropic Streaming API example with Revenium middleware.

This example demonstrates how to use Anthropic's streaming API with automatic
Revenium metering. The middleware tracks token usage and timing for streaming
responses just like regular API calls.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


def main():
    """Run streaming Anthropic API example."""

    print("Anthropic Streaming API Example with Revenium Middleware")
    print("=" * 65)

    # Import Anthropic and Revenium middleware
    import anthropic
    import revenium_middleware.anthropic
    from revenium_middleware.anthropic import usage_context

    # Demonstrate hybrid initialization for streaming
    print("Middleware Initialization:")
    if revenium_middleware.anthropic.is_initialized():
        print("   Middleware ready for streaming")
    else:
        print("   Initializing middleware for streaming...")
        if revenium_middleware.anthropic.initialize():
            print("   Streaming middleware initialized")
        else:
            print("   Initialization incomplete - streaming will continue")
    print()

    # Set metadata in context for streaming
    usage_context.set(
        {
            "subscriber": {
                "email": "ai@revenium.io",
                "credential": {"name": "content-api-key"},
            },
            "trace_id": "streaming-conv-001",
            "task_id": "task-41921",
            "task_type": "creative-writing",
            "agent": "anthropic-streaming-python",
            "organizationName": "AcmeCorp",
            "productName": "customer-chatbot",
        }
    )

    # Create Anthropic client (middleware is automatically applied)
    client = anthropic.Anthropic()

    try:
        print("Starting streaming response...")
        print("Response:")
        print("-" * 40)

        # Use streaming API with context manager
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=200,
            temperature=0.9,
            messages=[
                {
                    "role": "user",
                    "content": "Write a short poem about artificial intelligence and creativity.",
                }
            ],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)

        print("\n" + "-" * 40)
        print("Streaming completed!")

        # Get final message with token usage
        final_message = stream.get_final_message()
        print(
            f"Tokens: {final_message.usage.input_tokens} input + {final_message.usage.output_tokens} output"
        )
        print("Token usage automatically metered after stream completion")

        print("\nMetadata sent to Revenium:")
        print("   • Trace ID: streaming-conv-001")
        print("   • Task Type: creative-writing")
        print("   • Organization: AcmeCorp")
        print("   • Agent: content-generator")
        print("   • Product: customer-chatbot")

    except Exception as e:
        print(f"Error: {e}")
        return False

    print("\nStreaming Benefits:")
    print("   • Real-time response display")
    print("   • Better user experience for long responses")
    print("   • Automatic token counting and metering")
    print("   • Same metadata tracking as regular API calls")

    print("\nHybrid Initialization for Streaming:")
    print("   • Auto-initialization: Works seamlessly with streaming")
    print("   • Context management: usage_context.set() works with any init method")
    print("   • Graceful handling: Streaming continues even if metering fails")
    print("   • Explicit control: Call initialize() before streaming if needed")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
