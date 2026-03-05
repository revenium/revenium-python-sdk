#!/usr/bin/env python3
"""
Basic Anthropic API example with Revenium middleware.

This example shows the simplest way to use Anthropic's API with automatic
Revenium metering. Just import the middleware and use Anthropic normally.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def main():
    """Run basic Anthropic API example."""
    
    print("Basic Anthropic API Example with Revenium Middleware")
    print("=" * 60)

    # Import Anthropic and Revenium middleware
    import anthropic
    import revenium_middleware.anthropic

    # Check initialization status (middleware auto-initializes on import)
    if revenium_middleware.anthropic.is_initialized():
        print("Revenium middleware auto-initialized successfully")
    else:
        print("Warning: Revenium middleware not initialized - manual setup may be needed")
        # For explicit control, you could call:
        # success = revenium_middleware.anthropic.initialize()
        print("   (Continuing with example - middleware will still work)")
    print()

    # Create Anthropic client (middleware is automatically applied)
    client = anthropic.Anthropic()

    try:
        print("Making API call to Anthropic...")

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.7,
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
            ]
        )

        print("Request successful!")
        print(f"Response: {message.content[0].text}")
        print(f"Tokens: {message.usage.input_tokens} input + {message.usage.output_tokens} output")
        print("Automatically metered to Revenium!")

    except Exception as e:
        print(f"Error: {e}")
        return False

    print("\nHybrid Initialization Benefits:")
    print("   • Auto-initialization: Just import and use (simple experience)")
    print("   • Explicit control: Call initialize() for advanced configuration")
    print("   • Graceful fallback: No exceptions if setup incomplete")
    print("\nNote: This basic example uses automatic metering without custom metadata.")
    print("   For advanced tracking with custom metadata, see anthropic-advanced.py")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
