#!/usr/bin/env python3
"""
Advanced Anthropic API example with detailed Revenium metadata tracking.

This example shows how to use custom metadata for detailed tracking and analytics
in your Revenium dashboard. Perfect for production applications that need
granular usage tracking.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def main():
    """Run advanced Anthropic API example with detailed metadata."""
    
    print("Advanced Anthropic API Example with Detailed Metadata")
    print("=" * 65)

    # Import Anthropic and Revenium middleware
    import anthropic
    import revenium_middleware.anthropic

    # Demonstrate explicit initialization control
    print("Initialization Status:")
    if revenium_middleware.anthropic.is_initialized():
        print("   Auto-initialized successfully")
    else:
        print("    Auto-initialization incomplete")
        print("   Attempting explicit initialization...")
        success = revenium_middleware.anthropic.initialize()
        if success:
            print("   Explicit initialization successful")
        else:
            print("   Explicit initialization failed - check configuration")
    print()

    # Create Anthropic client (middleware is automatically applied)
    client = anthropic.Anthropic()

    try:
        print("Making API call with detailed metadata tracking...")

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            temperature=0.8,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the meaning of life, the universe and everything?",
                        }
                    ]
                }
            ],
            usage_metadata={
                "subscriber": {
                    "id": "subscriberid-1234567890",
                    "email": "user@example.com",
                    "credential": {
                        "name": "engineering-api-key",
                        "value": "sk-ant-api03-..."
                    }
                },
                "organizationName": "AcmeCorp",
                "subscription_id": "startup-plan-Q1",
                "productName": "customer-chatbot",
                "trace_id": "conv-28a7e9d4",
                "task_type": "summarize-customer-issue",
                "agent": "anthropic-python-advanced",
            }
        )

        print("Request successful!")
        print(f"Response: {message.content[0].text}")
        print(f"Tokens: {message.usage.input_tokens} input + {message.usage.output_tokens} output")
        print("Metered to Revenium with detailed metadata!")

        print("\nMetadata sent to Revenium:")
        print("   • Trace ID: conv-28a7e9d4")
        print("   • Task Type: summarize-customer-issue")
        print("   • Organization: AcmeCorp")
        print("   • Subscriber: user@example.com")
        print("   • Agent: support-agent")
        print("   • Product: customer-chatbot")

    except Exception as e:
        print(f"Error: {e}")
        return False

    print("\nBenefits of detailed metadata:")
    print("   • Track usage by customer, team, or product")
    print("   • Analyze costs per feature or user")
    print("   • Monitor AI agent performance")
    print("   • Enable detailed billing and chargeback")

    print("\nInitialization Options:")
    print("   • Auto-initialization: Middleware activates on import")
    print("   • Explicit control: Call initialize() for custom setup")
    print("   • Status checking: Use is_initialized() to verify state")
    print("   • Graceful fallback: No exceptions on configuration issues")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
