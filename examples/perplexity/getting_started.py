"""
Getting Started Example

This example demonstrates the simplest way to use the Revenium Perplexity middleware.
Just import the middleware and use the OpenAI client with Perplexity's base URL.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import Revenium middleware (this automatically patches OpenAI)
import revenium_middleware.perplexity  # noqa: F401

def main():
    """Run a simple Perplexity chat completion with automatic metering."""
    
    # Create OpenAI client with Perplexity base URL
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    print("Sending request to Perplexity...")
    
    # Make a chat completion request
    response = client.chat.completions.create(
        model="sonar-pro",
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

