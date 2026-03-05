"""
Basic Example

This example shows how to use the middleware with custom metadata
to track business context like organization, product, and subscriber information.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import Revenium middleware
import revenium_middleware.perplexity  # noqa: F401


def main():
    """Run a Perplexity chat completion with custom metadata."""

    # Create OpenAI client with Perplexity base URL
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    print("Sending request to Perplexity with custom metadata...")

    # Make a chat completion request with usage metadata
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {
                "role": "user",
                "content": "Explain what Perplexity AI is in 2-3 sentences."
            }
        ],
        # Custom metadata for tracking
        usage_metadata={
            "organization_id": "org-demo-123",
            "product_id": "prod-perplexity-demo",
            "subscription_id": "sub-premium",
            "subscriber": {
                "id": "user-456",
                "email": "demo@example.com"
            },
            "task_type": "chat",
            "trace_id": "demo-trace-001"
        }
    )

    # Display the response
    print(f"\nAssistant: {response.choices[0].message.content}")
    print("\nTokens used:")
    print(f"  - Input: {response.usage.prompt_tokens}")
    print(f"  - Output: {response.usage.completion_tokens}")
    print(f"  - Total: {response.usage.total_tokens}")
    print("\n✅ Usage data with metadata sent to Revenium!")


if __name__ == "__main__":
    main()
