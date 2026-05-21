"""
Example demonstrating decorator support for automatic metadata injection.

This example shows how to use the @revenium_metadata decorator to automatically
inject metadata into all OpenAI API calls within a function, without having to
pass usage_metadata to each individual call.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

import revenium_middleware.openai.middleware
from revenium_middleware import revenium_metadata, revenium_meter


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@revenium_metadata(
    trace_id="session-12345",
    task_type="customer-support",
    organization_name="AcmeCorp"
)
def handle_customer_query(question: str) -> str:
    """
    Handle a customer query with automatic metadata injection.

    All OpenAI calls within this function will automatically include:
    - trace_id: "session-12345"
    - task_type: "customer-support"
    - organization_name: "AcmeCorp"
    """
    print(f"\nProcessing query: {question}")

    # This call automatically includes the decorator metadata
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful customer support assistant."
            },
            {"role": "user", "content": question}
        ],
        max_tokens=100
    )

    answer = response.choices[0].message.content
    print(f"Answer: {answer}")

    return answer


@revenium_metadata(
    trace_id="batch-process-001",
    task_type="data-analysis"
)
def analyze_multiple_queries(queries: list) -> list:
    """
    Process multiple queries with shared metadata.

    All queries in this batch will share the same trace_id and task_type.
    """
    print(f"\nAnalyzing {len(queries)} queries...")
    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n  Query {i}/{len(queries)}: {query}")

        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": query}],
            max_tokens=50
        )

        result = response.choices[0].message.content
        results.append(result)
        print(f"  Result: {result}")

    return results


@revenium_metadata(
    trace_id="embedding-session",
    task_type="semantic-search"
)
def create_embeddings(texts: list) -> list:
    """
    Create embeddings with automatic metadata injection.

    Demonstrates that decorators work with embeddings API too.
    """
    print(f"\nCreating embeddings for {len(texts)} texts...")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    print(f"Created {len(embeddings)} embeddings")

    return embeddings


@revenium_meter()
@revenium_metadata(
    trace_id="selective-metering-demo",
    task_type="premium-feature"
)
def premium_feature(prompt: str) -> str:
    """
    Demonstrates combining @revenium_meter with @revenium_metadata.

    When REVENIUM_SELECTIVE_METERING=true, only calls within functions
    decorated with @revenium_meter will be metered.
    """
    print(f"\nPremium feature processing: {prompt}")

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    result = response.choices[0].message.content
    print(f"Premium result: {result}")

    return result


def nested_decorator_example():
    """
    Demonstrates nested decorators with different metadata.
    """
    print("\nNested decorator example:")

    @revenium_metadata(
        trace_id="outer-trace",
        task_type="outer-task",
        organization_name="OuterOrg"
    )
    def outer_function():
        print("   Outer function called")

        @revenium_metadata(
            trace_id="inner-trace",  # This overrides outer
            task_type="inner-task"    # This overrides outer
            # organization_name not specified, so outer value is used
        )
        def inner_function():
            print("   Inner function called")
            response = client.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=20
            )
            return response.choices[0].message.content

        return inner_function()

    result = outer_function()
    print(f"Result: {result}")
    print("   Metadata used:")
    print("   - trace_id: 'inner-trace' (from inner decorator)")
    print("   - task_type: 'inner-task' (from inner decorator)")
    print("   - organization_name: 'OuterOrg' (from outer decorator)")


def main():
    """Run all decorator examples."""
    print("=" * 70)
    print("OpenAI Middleware - Decorator Support Examples")
    print("=" * 70)

    # Example 1: Single query with metadata
    print("\n" + "=" * 70)
    print("Example 1: Single Query with Automatic Metadata")
    print("=" * 70)
    handle_customer_query("How do I reset my password?")

    # Example 2: Batch processing with shared metadata
    print("\n" + "=" * 70)
    print("Example 2: Batch Processing with Shared Metadata")
    print("=" * 70)
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is TypeScript?"
    ]
    analyze_multiple_queries(queries)

    # Example 3: Embeddings with metadata
    print("\n" + "=" * 70)
    print("Example 3: Embeddings with Metadata")
    print("=" * 70)
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing handles text"
    ]
    create_embeddings(texts)

    # Example 4: Combined decorators
    print("\n" + "=" * 70)
    print("Example 4: Combined Decorators")
    print("=" * 70)
    premium_feature("Analyze this premium content")

    # Example 5: Nested decorators
    print("\n" + "=" * 70)
    print("Example 5: Nested Decorators")
    print("=" * 70)
    nested_decorator_example()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("   1. Use @revenium_metadata to inject metadata automatically")
    print("   2. Decorators work with chat, embeddings, and streaming")
    print("   3. Inner decorators override outer decorator metadata")
    print("   4. Combine @revenium_meter for selective metering")
    print("   5. All metering data sent to Revenium automatically")
    print("\nCheck your Revenium dashboard to see the metered usage!")


if __name__ == "__main__":
    main()
