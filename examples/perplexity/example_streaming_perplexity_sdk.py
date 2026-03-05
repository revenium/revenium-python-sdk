#!/usr/bin/env python3
"""
Example: Streaming with Native Perplexity SDK

This example demonstrates streaming responses using the native Perplexity SDK
with Revenium metering.
"""
import os
from dotenv import load_dotenv
from perplexity import Perplexity

# Load environment variables from .env file
load_dotenv()

# Import Revenium middleware - this automatically patches the Perplexity SDK
import revenium_middleware.perplexity


def main():
    """Run a streaming Perplexity chat completion using native SDK."""
    
    # Create Perplexity client
    client = Perplexity(
        api_key=os.getenv("PERPLEXITY_API_KEY")
    )
    
    print("Sending streaming request to Perplexity via native SDK...")
    print("\nAssistant: ", end="", flush=True)
    
    # Make a streaming chat completion request
    stream = client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "user",
                "content": "Explain quantum computing in 2-3 sentences."
            }
        ],
        stream=True
    )
    
    # Process the stream
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n\n✅ Usage data automatically sent to Revenium!")


if __name__ == "__main__":
    main()

