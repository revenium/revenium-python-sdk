"""
Example demonstrating decorator-based metadata injection with Anthropic middleware.

This example shows how to use @revenium_metadata and @revenium_meter decorators
to automatically inject metadata into Anthropic API calls without passing it explicitly.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Anthropic middleware FIRST to register wrappers
import revenium_middleware.anthropic
from anthropic import Anthropic
from revenium_middleware import revenium_metadata, revenium_meter

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@revenium_metadata(
    trace_id="session-12345",
    task_type="customer-support",
    organizationName="AcmeCorp",
    user_id="user-789"
)
def handle_customer_query(question: str) -> str:
    """
    Example 1: Single query with automatic metadata injection.
    
    All Anthropic API calls within this function will automatically include
    the metadata defined in the decorator.
    """
    print("\n=== Example 1: Single Query with Decorator ===")
    print(f"Question: {question}")
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    
    answer = response.content[0].text
    print(f"Answer: {answer}")
    print("✅ Metadata automatically included: trace_id, task_type, organizationName, user_id")
    
    return answer


@revenium_metadata(
    trace_id="batch-process-001",
    task_type="data-analysis",
    organizationName="AnalyticsTeam"
)
def analyze_multiple_queries(queries: list) -> list:
    """
    Example 2: Batch processing with shared metadata.
    
    All queries in the batch share the same metadata from the decorator.
    """
    print("\n=== Example 2: Batch Processing with Shared Metadata ===")
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nProcessing query {i}/{len(queries)}: {query}")
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        answer = response.content[0].text
        results.append(answer)
        print(f"Result: {answer}")
    
    print(f"\n✅ All {len(queries)} queries metered with shared metadata")
    return results


@revenium_meter
@revenium_metadata(
    trace_id="premium-feature-001",
    feature_tier="premium",
    organizationName="PremiumCustomer"
)
def premium_feature(prompt: str) -> str:
    """
    Example 3: Combined decorators for selective metering.
    
    @revenium_meter: Only meters when REVENIUM_SELECTIVE_METERING=true
    @revenium_metadata: Adds metadata to the metered call
    """
    print("\n=== Example 3: Combined Decorators (Selective Metering) ===")
    print(f"Prompt: {prompt}")
    print("Note: This is only metered if REVENIUM_SELECTIVE_METERING=true")
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.content[0].text
    print(f"Result: {result}")
    print("✅ Metered with premium feature metadata")
    
    return result


@revenium_metadata(
    trace_id="outer-context",
    organizationName="OuterOrg",
    task_type="outer-task"
)
def nested_decorator_example():
    """
    Example 4: Nested decorators showing metadata override.
    
    Inner decorator metadata takes precedence over outer decorator metadata.
    """
    print("\n=== Example 4: Nested Decorators (Metadata Override) ===")
    
    # This call uses outer decorator metadata
    print("\nOuter context call:")
    response1 = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ]
    )
    print(f"Answer: {response1.content[0].text}")
    print("Metadata: trace_id=outer-context, organizationName=OuterOrg, task_type=outer-task")
    
    # Inner function with its own decorator
    @revenium_metadata(
        trace_id="inner-context",  # Overrides outer trace_id
        user_id="inner-user"       # Adds new field
        # organizationName and task_type inherited from outer
    )
    def inner_function():
        print("\nInner context call:")
        response2 = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "What is 3+3?"}
            ]
        )
        print(f"Answer: {response2.content[0].text}")
        print("Metadata: trace_id=inner-context (overridden), organizationName=OuterOrg (inherited), user_id=inner-user (new)")
        return response2
    
    inner_function()
    print("\n✅ Nested decorators demonstrate metadata inheritance and override")


@revenium_metadata(
    trace_id="streaming-session",
    task_type="streaming-chat",
    organizationName="StreamingOrg"
)
def streaming_with_decorator(prompt: str):
    """
    Example 5: Streaming with decorator metadata.
    
    Decorators work with streaming responses too.
    """
    print("\n=== Example 5: Streaming with Decorator ===")
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)
    
    with client.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {"role": "user", "content": prompt}
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    
    print("\n✅ Streaming response metered with decorator metadata")


def main():
    """Run all decorator examples."""
    print("=" * 70)
    print("Anthropic Middleware - Decorator Examples")
    print("=" * 70)
    
    # Example 1: Single query
    handle_customer_query("What are your business hours?")
    
    # Example 2: Batch processing
    queries = [
        "What is Python?",
        "What is AI?",
        "What is cloud computing?"
    ]
    analyze_multiple_queries(queries)
    
    # Example 3: Combined decorators
    premium_feature("Explain quantum computing in simple terms")
    
    # Example 4: Nested decorators
    nested_decorator_example()
    
    # Example 5: Streaming
    streaming_with_decorator("Tell me a short joke")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. ✅ @revenium_metadata automatically injects metadata into API calls")
    print("2. ✅ @revenium_meter enables selective metering for specific functions")
    print("3. ✅ Decorators can be combined for powerful metering control")
    print("4. ✅ Nested decorators allow metadata override and inheritance")
    print("5. ✅ Works with both regular and streaming API calls")
    print("\nCheck your Revenium dashboard to see the metered data!")


if __name__ == "__main__":
    main()

