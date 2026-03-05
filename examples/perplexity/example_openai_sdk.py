#!/usr/bin/env python3
"""
Example: Using Revenium Middleware with OpenAI SDK

This example demonstrates how to use the Revenium middleware with the
OpenAI SDK configured to use Perplexity's API endpoint.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Import Revenium middleware - this automatically patches the OpenAI SDK
import revenium_middleware.perplexity


def main():
    """Run a simple Perplexity chat completion using OpenAI SDK."""
    
    # Create OpenAI client with Perplexity base URL
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    print("Sending request to Perplexity via OpenAI SDK...")
    
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

