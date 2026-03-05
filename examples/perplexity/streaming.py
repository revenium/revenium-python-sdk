"""
Streaming Example

This example demonstrates how to use streaming responses with the middleware.
The middleware automatically collects usage data from the final stream chunk.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import Revenium middleware
import revenium_middleware.perplexity  # noqa: F401

def main():
    """Run a streaming Perplexity chat completion with automatic metering."""
    
    # Create OpenAI client with Perplexity base URL
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    print("Streaming response from Perplexity...\n")
    
    # Make a streaming chat completion request
    stream = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {
                "role": "user",
                "content": "Write a short poem about artificial intelligence."
            }
        ],
        stream=True,
        # Optional: Add custom metadata
        usage_metadata={
            "organization_id": "org-streaming-demo",
            "product_id": "prod-perplexity-stream",
            "task_type": "creative_writing"
        }
    )
    
    # Process the stream
    print("Assistant: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n\n✅ Streaming complete! Usage data sent to Revenium!")


if __name__ == "__main__":
    main()

