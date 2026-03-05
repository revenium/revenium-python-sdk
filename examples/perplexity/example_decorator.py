#!/usr/bin/env python3
"""
Example: Using Decorators for Automatic Metadata Injection

This example demonstrates how to use the @revenium_metadata decorator to
automatically inject metadata into all Perplexity API calls within a function,
eliminating the need to pass usage_metadata to each individual call.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import middleware and decorators
import revenium_middleware.perplexity
from revenium_middleware.perplexity import revenium_metadata, revenium_meter


# Initialize OpenAI client with Perplexity base URL
client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)


@revenium_metadata(
    organization_id="acme-corp",
    task_type="customer-support",
    trace_id="session-12345"
)
def handle_customer_query(question: str) -> str:
    """
    Handle a customer query with automatic metadata injection.

    All Perplexity calls within this function will automatically include:
    - organization_id: "acme-corp"
    - task_type: "customer-support"
    - trace_id: "session-12345"
    """
    print(f"\n{'='*60}")
    print(f"Processing customer query: {question}")
    print(f"{'='*60}")

    # This call automatically includes the decorator metadata
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful customer support assistant."
            },
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    print(f"\nAnswer: {answer}")
    print(f"Tokens used: {response.usage.total_tokens}")

    return answer


@revenium_metadata(
    organization_id="acme-corp",
    task_type="data-analysis",
    product_id="analytics-pro"
)
def analyze_multiple_topics(topics: list) -> dict:
    """
    Analyze multiple topics with automatic metadata injection.

    Demonstrates that metadata is applied to ALL calls within the function.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {len(topics)} topics...")
    print(f"{'='*60}")

    results = {}

    for topic in topics:
        print(f"\nAnalyzing: {topic}")

        # Each call gets the same metadata automatically
        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {
                    "role": "user",
                    "content": f"Provide a brief analysis of: {topic}"
                }
            ]
        )

        results[topic] = response.choices[0].message.content
        print(f"  ✓ Completed")

    return results


@revenium_metadata(
    organization_id="acme-corp",
    task_type="default"
)
def mixed_metadata_example(query: str) -> str:
    """
    Demonstrates API-level metadata override.

    API-level metadata takes precedence over decorator metadata.
    """
    print(f"\n{'='*60}")
    print("Mixed metadata example")
    print(f"{'='*60}")

    # This call uses decorator metadata (task_type="default")
    print("\n1. Using decorator metadata...")
    response1 = client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": query}]
    )

    # This call overrides with API-level metadata (task_type="special")
    print("2. Overriding with API-level metadata...")
    response2 = client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": query}],
        usage_metadata={
            "task_type": "special-override"
        }
    )

    print(f"\nBoth calls completed successfully!")
    return response1.choices[0].message.content


@revenium_meter()
@revenium_metadata(
    organization_id="premium-corp",
    task_type="premium-feature",
    subscription_tier="enterprise"
)
def premium_feature(prompt: str) -> str:
    """
    Demonstrates combining @revenium_meter with @revenium_metadata.

    When REVENIUM_SELECTIVE_METERING=true, only calls within functions
    decorated with @revenium_meter will be metered.
    """
    print(f"\n{'='*60}")
    print("Premium feature (selective metering)")
    print(f"{'='*60}")

    response = client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.choices[0].message.content
    print(f"\nPremium result: {result}")

    return result


def main():
    """Run all decorator examples."""
    print("\n" + "="*60)
    print("Decorator Examples for Perplexity Middleware")
    print("="*60)

    # Example 1: Basic metadata injection
    handle_customer_query("How do I reset my password?")

    # Example 2: Multiple calls with same metadata
    topics = ["Artificial Intelligence", "Quantum Computing"]
    analyze_multiple_topics(topics)

    # Example 3: API-level override
    mixed_metadata_example("What is machine learning?")

    # Example 4: Selective metering (if enabled)
    premium_feature("Explain neural networks")

    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("Check your Revenium dashboard to see the metered calls")
    print("="*60)


if __name__ == "__main__":
    main()

