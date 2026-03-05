#!/usr/bin/env python3
"""
Example: Using Revenium Middleware with Native Perplexity SDK

This example demonstrates how to use the Revenium middleware with the
native Perplexity Python SDK (perplexityai package).
"""
import os
from dotenv import load_dotenv
from perplexity import Perplexity

# Load environment variables from .env file
load_dotenv()

# Import Revenium middleware - this automatically patches the Perplexity SDK
import revenium_middleware.perplexity


def main():
    """Run a simple Perplexity chat completion using native SDK."""
    
    # Create Perplexity client
    client = Perplexity(
        api_key=os.getenv("PERPLEXITY_API_KEY")
    )
    
    print("Sending request to Perplexity via native SDK...")
    
    # Make a chat completion request
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one sentence."
            }
        ]
    )
    
    # Display the response
    print(f"\nAssistant: {response.choices[0].message.content}")
    print(f"\nTokens used: {response.usage.total_tokens}")
    print("\n✅ Usage data automatically sent to Revenium!")


if __name__ == "__main__":
    main()

