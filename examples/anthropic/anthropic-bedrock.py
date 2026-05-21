#!/usr/bin/env python3
"""
Comprehensive AWS Bedrock Example: Revenium Middleware for Anthropic

This example demonstrates complete AWS Bedrock integration with the Revenium middleware:
1. Basic chat completion via AWS Bedrock
2. Chat completion with metadata tracking via AWS Bedrock
3. Bedrock streaming support
4. Advanced Bedrock streaming with detailed usage tracking
5. Model mapping and configuration options

Prerequisites:
- Install: pip install revenium-python-sdk[anthropic]
- Set REVENIUM_METERING_API_KEY environment variable
- Configure AWS credentials with bedrock:InvokeModel and bedrock:InvokeModelWithResponseStream permissions
- Ensure Claude models are available in your AWS region
"""

import os
from dotenv import load_dotenv
import anthropic
import revenium_middleware.anthropic
from revenium_middleware.anthropic import usage_context

load_dotenv()  # Load environment variables from .env file


def example_1_basic_chat():
    """Example 1: Basic chat completion using AWS Bedrock."""

    print("Example 1: Basic Chat Completion (AWS Bedrock)")
    print("-" * 60)

    # Force Bedrock routing with explicit base_url
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bedrock_base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com"

    print(f"Using Bedrock base_url: {bedrock_base_url}")
    client = anthropic.Anthropic(base_url=bedrock_base_url)

    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "Hello! What is AI?"}],
            max_tokens=50,
        )

        if response and hasattr(response, 'content') and hasattr(response, 'usage'):
            print("Bedrock request successful!")
            print(f"Response: {response.content[0].text}")
            print(
                f"Tokens: {response.usage.input_tokens} input + {response.usage.output_tokens} output"
            )
            print("Automatically metered to Revenium with provider='AWS'")
        else:
            print("Warning: Bedrock request returned but response format unexpected")
            print(f"Response: {response}")

    except Exception as e:
        print(f"Bedrock error: {e}")
        print(
            "Ensure AWS credentials are configured with bedrock:InvokeModel permission"
        )


def example_2_chat_with_metadata():
    """Example 2: Bedrock chat completion with detailed metadata tracking."""

    print("\nExample 2: Bedrock Chat with Metadata Tracking")
    print("-" * 60)

    # Force Bedrock routing with explicit base_url
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bedrock_base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com"

    print(f"Using Bedrock base_url: {bedrock_base_url}")
    client = anthropic.Anthropic(base_url=bedrock_base_url)

    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "Explain machine learning briefly."}],
            max_tokens=75,
            usage_metadata={
                "subscriber": {"email": "demo@example.com"},
                "trace_id": "demo-conversation-001",
                "task_type": "educational-explanation",
                "organizationName": "AcmeCorp",
                "agent": "anthropic-python-metadata",
            },
        )

        if response and hasattr(response, 'content') and hasattr(response, 'usage'):
            print("Bedrock request successful!")
            print(f"Response: {response.content[0].text}")
            print(
                f"Tokens: {response.usage.input_tokens} input + {response.usage.output_tokens} output"
            )
            print("Metered to Revenium with provider='AWS' and detailed metadata")
        else:
            print("Warning: Bedrock request returned but response format unexpected")
            print(f"Response: {response}")

    except Exception as e:
        print(f"Bedrock error: {e}")
        print(
            "Ensure AWS credentials are configured with bedrock:InvokeModel permission"
        )


def example_3_bedrock_integration():
    """Example 3: AWS Bedrock integration with explicit base_url."""

    print("\nExample 3: AWS Bedrock Integration")
    print("-" * 60)

    # Force Bedrock routing with explicit base_url
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bedrock_base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com"

    print(f"AWS Region: {aws_region}")
    print(f"Using Bedrock base_url: {bedrock_base_url}")

    # Force Bedrock routing with explicit base_url
    client = anthropic.Anthropic(base_url=bedrock_base_url)

    try:
        response = client.messages.create(
            model="claude-opus-4-7",  # Automatically maps to Bedrock model ID
            messages=[{"role": "user", "content": "What is AWS Bedrock?"}],
            max_tokens=100,
            usage_metadata={
                "trace_id": "bedrock-demo-001",
                "task_type": "bedrock-explanation",
                "organizationName": "AcmeCorp",
            },
        )

        if response and hasattr(response, 'content') and hasattr(response, 'usage'):
            print("Bedrock request successful!")
            print(f"Response: {response.content[0].text}")
            print(
                f"Tokens: {response.usage.input_tokens} input + {response.usage.output_tokens} output"
            )
            print("Automatically metered to Revenium with provider='AWS'")
        else:
            print("Warning: Bedrock request returned but response format unexpected")
            print(f"Response: {response}")

    except Exception as e:
        print(f"Bedrock error: {e}")
        print(
            "Ensure AWS credentials are configured with bedrock:InvokeModel permission"
        )


def example_4_streaming():
    """Example 4: Bedrock streaming support."""

    print("\nExample 4: Bedrock Streaming Support")
    print("-" * 60)
    print("Demonstrating streaming with AWS Bedrock!")

    # Set metadata in context for streaming
    usage_context.set(
        {
            "trace_id": "bedrock-streaming-demo-001",
            "task_type": "bedrock-streaming-chat",
            "organizationName": "AcmeCorp",
        }
    )

    # Force Bedrock routing with explicit base_url
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bedrock_base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com"

    print(f"Using Bedrock base_url: {bedrock_base_url}")
    client = anthropic.Anthropic(base_url=bedrock_base_url)

    try:
        print("Starting streaming response...")
        with client.messages.stream(
            model="claude-opus-4-7",
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
            max_tokens=50,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)

        print("\nBedrock streaming completed!")
        print("Token usage automatically metered after stream completion")

        # Get final message with usage information
        final_message = stream.get_final_message()
        if final_message and hasattr(final_message, 'usage'):
            print(
                f"Tokens: {final_message.usage.input_tokens} input + {final_message.usage.output_tokens} output"
            )
            print("Routed through AWS Bedrock streaming")
            print("Automatically metered to Revenium with provider='AWS'")
        else:
            print("Warning: Streaming completed but final message format unexpected")
            print(f"Final message: {final_message}")

    except Exception as e:
        print(f"Bedrock streaming error: {e}")
        print(
            "Ensure AWS credentials are configured with bedrock:InvokeModelWithResponseStream permission"
        )


def example_5_bedrock_streaming():
    """Example 5: Dedicated Bedrock streaming example."""

    print("\nExample 5: Bedrock Streaming (Advanced)")
    print("-" * 60)

    # Force Bedrock detection by using base_url
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bedrock_base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com"

    print(f"Using Bedrock base_url: {bedrock_base_url}")

    # Set metadata for detailed tracking
    usage_context.set(
        {
            "trace_id": "bedrock-streaming-demo-001",
            "task_type": "bedrock-streaming-test",
            "organizationName": "AcmeCorp",
            "agent": "anthropic-python-bedrock-streaming",
        }
    )

    client = anthropic.Anthropic(base_url=bedrock_base_url)

    try:
        print("Starting Bedrock streaming response...")
        with client.messages.stream(
            model="claude-opus-4-7",
            messages=[
                {"role": "user", "content": "Write a haiku about streaming data."}
            ],
            max_tokens=100,
        ) as stream:
            print("Response: ", end="")
            for text in stream.text_stream:
                print(text, end="", flush=True)

        print("\nBedrock streaming completed!")

        # Get final message with detailed usage information
        final_message = stream.get_final_message()
        if final_message and hasattr(final_message, 'usage'):
            print(f"Token Usage:")
            print(f"   Input tokens: {final_message.usage.input_tokens}")
            print(f"   Output tokens: {final_message.usage.output_tokens}")
            print(f"   Total tokens: {final_message.usage.total_tokens}")
            print(f"Automatically metered to Revenium with provider='AWS'")
        else:
            print("Warning: Streaming completed but final message format unexpected")
            print(f"Final message: {final_message}")

    except Exception as e:
        print(f"Bedrock streaming error: {e}")
        print("This is normal if AWS credentials aren't configured")
        print("   The middleware will fall back to direct Anthropic API")


def demonstrate_model_mapping():
    """Show how different Anthropic models map to Bedrock model IDs."""

    print("\nModel Mapping Examples:")
    print("The middleware automatically maps Anthropic model names to Bedrock IDs:")

    model_mappings = {
        # claude-opus-4-7 — region variants (verified 2026-05-12)
        "claude-opus-4-7": "anthropic.claude-opus-4-7",
        "claude-opus-4-7 (us)": "us.anthropic.claude-opus-4-7",
        "claude-opus-4-7 (eu)": "eu.anthropic.claude-opus-4-7",
        "claude-opus-4-7 (au)": "au.anthropic.claude-opus-4-7",
        "claude-opus-4-7 (global)": "global.anthropic.claude-opus-4-7",
        # claude-3-5 series
        "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
    }

    for anthropic_model, bedrock_id in model_mappings.items():
        print(f"   • {anthropic_model} → {bedrock_id}")


def demonstrate_configuration():
    """Show configuration options for Bedrock integration."""

    print("\nConfiguration Options:")
    print("Environment variables for Bedrock integration:")
    print(f"   • AWS_REGION: {os.getenv('AWS_REGION', 'us-east-1 (default)')}")
    print(
        f"   • REVENIUM_BEDROCK_DISABLE: {os.getenv('REVENIUM_BEDROCK_DISABLE', 'Not set (Bedrock enabled)')}"
    )
    print(
        f"   • REVENIUM_METERING_API_KEY: {'Set' if os.getenv('REVENIUM_METERING_API_KEY') else 'Not set'}"
    )

    print("\nAWS Authentication methods (in order of precedence):")
    print("   1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("   2. AWS credentials file (~/.aws/credentials)")
    print("   3. IAM roles (for EC2/Lambda/ECS)")
    print("   4. AWS SSO")

    print("\nRequired AWS permissions:")
    print("   • bedrock:InvokeModel (for non-streaming requests)")
    print("   • bedrock:InvokeModelWithResponseStream (for streaming requests)")

    print("\nStreaming Support:")
    print("   • Works with both Anthropic API and AWS Bedrock")
    print("   • Automatic provider detection and routing")
    print("   • Identical interface regardless of provider")
    print("   • Graceful fallback if Bedrock streaming fails")

    print("\nHybrid Initialization:")
    print("   • Auto-initialization: Middleware activates on import")
    print("   • Explicit control: Call initialize() for custom configuration")
    print("   • Status checking: Use is_initialized() to verify setup")
    print("   • Graceful fallback: No exceptions on configuration issues")
    print("   • Works with all providers: Anthropic direct and AWS Bedrock")


def main():
    """Run all AWS Bedrock examples to demonstrate complete Bedrock integration."""

    print("Revenium Middleware for Anthropic - AWS Bedrock Integration Examples")
    print("=" * 75)

    # Import middleware and check initialization
    import revenium_middleware.anthropic

    print("Middleware Initialization Status:")
    if revenium_middleware.anthropic.is_initialized():
        print("   Auto-initialization successful")
    else:
        print("   Auto-initialization incomplete, trying explicit initialization...")
        if revenium_middleware.anthropic.initialize():
            print("   Explicit initialization successful")
        else:
            print(
                "   Initialization incomplete - examples will continue with fallback"
            )
    print()

    # Check environment setup
    revenium_key = os.getenv("REVENIUM_METERING_API_KEY")
    if not revenium_key:
        print("Warning: REVENIUM_METERING_API_KEY not set")
        print("   Add it to your .env file: REVENIUM_METERING_API_KEY=\"hak_...\"")
        print("   Examples will still run but won't meter to Revenium\n")

    # Run all examples
    example_1_basic_chat()
    example_2_chat_with_metadata()
    example_3_bedrock_integration()
    example_4_streaming()
    example_5_bedrock_streaming()

    # Show additional information
    demonstrate_model_mapping()
    demonstrate_configuration()


if __name__ == "__main__":
    main()

    print("\nAll AWS Bedrock examples complete!")
    print(
        "Check your Revenium dashboard to see all Bedrock usage metered with provider='AWS'"
    )
    print(
        "For more details, see: https://github.com/revenium/revenium-python-sdk"
    )
    print(
        "\nNote: All examples in this file route through AWS Bedrock for comprehensive testing:"
    )
    print("   • Basic chat completion via Bedrock")
    print("   • Metadata tracking via Bedrock")
    print("   • Streaming via Bedrock")
    print("   • Advanced streaming with detailed usage tracking via Bedrock")
