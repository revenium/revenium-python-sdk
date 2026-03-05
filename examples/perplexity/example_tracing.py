#!/usr/bin/env python3
"""
Example: Distributed Tracing with Perplexity Middleware

This example demonstrates how to use trace visualization fields for
distributed tracing and analytics. These fields help you track requests
across services and understand the context of API calls.
"""
import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import middleware
import revenium_middleware.perplexity


def setup_trace_context(
    environment="production",
    region="us-east-1",
    trace_type="customer-workflow",
    trace_name="Customer Onboarding Flow"
):
    """
    Set up trace context using environment variables.
    
    These environment variables are automatically picked up by the middleware
    and included in all metering data.
    """
    os.environ["REVENIUM_ENVIRONMENT"] = environment
    os.environ["REVENIUM_REGION"] = region
    os.environ["REVENIUM_TRACE_TYPE"] = trace_type
    os.environ["REVENIUM_TRACE_NAME"] = trace_name
    
    print(f"\n{'='*60}")
    print("Trace Context Configured:")
    print(f"  Environment: {environment}")
    print(f"  Region: {region}")
    print(f"  Trace Type: {trace_type}")
    print(f"  Trace Name: {trace_name}")
    print(f"{'='*60}")


def example_simple_trace():
    """
    Example 1: Simple trace with basic context.
    """
    print("\n" + "="*60)
    print("Example 1: Simple Trace")
    print("="*60)
    
    # Set up trace context
    setup_trace_context(
        environment="development",
        region="us-west-2",
        trace_type="test-workflow",
        trace_name="Simple Test"
    )
    
    # Create client
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    # Make a request - trace fields automatically included
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "user", "content": "What is distributed tracing?"}
        ]
    )
    
    print(f"\nResponse: {response.choices[0].message.content[:100]}...")
    print(f"Tokens: {response.usage.total_tokens}")
    print("✓ Trace data automatically sent to Revenium")


def example_distributed_trace():
    """
    Example 2: Distributed trace across multiple services.
    
    This simulates a request flowing through multiple services,
    with parent-child transaction relationships.
    """
    print("\n" + "="*60)
    print("Example 2: Distributed Trace")
    print("="*60)
    
    # Generate a trace ID for the entire workflow
    workflow_trace_id = str(uuid.uuid4())
    
    # Set up trace context for the workflow
    setup_trace_context(
        environment="production",
        region="us-east-1",
        trace_type="multi-service-workflow",
        trace_name="Document Processing Pipeline"
    )
    
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    # Step 1: Document Analysis Service
    print("\nStep 1: Document Analysis Service")
    os.environ["REVENIUM_TRANSACTION_NAME"] = "analyze-document"
    os.environ["REVENIUM_PARENT_TRANSACTION_ID"] = workflow_trace_id
    
    response1 = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "user", "content": "Analyze this document structure"}
        ],
        extra_body={
            "usage_metadata": {
                "service_name": "document-analyzer",
                "step": "1"
            }
        }
    )
    print(f"  ✓ Analysis complete ({response1.usage.total_tokens} tokens)")
    
    # Step 2: Content Extraction Service
    print("\nStep 2: Content Extraction Service")
    os.environ["REVENIUM_TRANSACTION_NAME"] = "extract-content"
    
    response2 = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "user", "content": "Extract key information"}
        ],
        extra_body={
            "usage_metadata": {
                "service_name": "content-extractor",
                "step": "2"
            }
        }
    )
    print(f"  ✓ Extraction complete ({response2.usage.total_tokens} tokens)")
    
    # Step 3: Summary Generation Service
    print("\nStep 3: Summary Generation Service")
    os.environ["REVENIUM_TRANSACTION_NAME"] = "generate-summary"
    
    response3 = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "user", "content": "Generate a summary"}
        ],
        extra_body={
            "usage_metadata": {
                "service_name": "summary-generator",
                "step": "3"
            }
        }
    )
    print(f"  ✓ Summary complete ({response3.usage.total_tokens} tokens)")
    
    print(f"\n✓ Distributed trace complete!")
    print(f"  Workflow Trace ID: {workflow_trace_id}")
    print(f"  Total tokens: {response1.usage.total_tokens + response2.usage.total_tokens + response3.usage.total_tokens}")


def example_retry_tracking():
    """
    Example 3: Tracking retries with trace fields.

    This shows how to track retry attempts for failed operations.
    """
    print("\n" + "="*60)
    print("Example 3: Retry Tracking")
    print("="*60)

    setup_trace_context(
        environment="production",
        region="us-east-1",
        trace_type="api-call-with-retry",
        trace_name="Resilient API Call"
    )

    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    max_retries = 3

    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1}/{max_retries}")

        # Set retry number in environment
        os.environ["REVENIUM_RETRY_NUMBER"] = str(attempt)
        os.environ["REVENIUM_TRANSACTION_NAME"] = f"api-call-attempt-{attempt + 1}"

        try:
            response = client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "user", "content": "What is resilience in distributed systems?"}
                ],
                extra_body={
                    "usage_metadata": {
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    }
                }
            )

            print(f"  ✓ Success! ({response.usage.total_tokens} tokens)")
            print(f"  Response: {response.choices[0].message.content[:80]}...")
            break

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            if attempt == max_retries - 1:
                print("  Maximum retries reached")
                raise


def example_multi_environment():
    """
    Example 4: Tracking across multiple environments.

    This shows how to differentiate between dev, staging, and production.
    """
    print("\n" + "="*60)
    print("Example 4: Multi-Environment Tracking")
    print("="*60)

    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    environments = ["development", "staging", "production"]

    for env in environments:
        print(f"\nTesting in {env.upper()} environment...")

        setup_trace_context(
            environment=env,
            region="us-east-1",
            trace_type="environment-test",
            trace_name=f"{env.capitalize()} Environment Test"
        )

        os.environ["REVENIUM_TRANSACTION_NAME"] = f"{env}-test"

        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "user", "content": f"Test query for {env}"}
            ],
            extra_body={
                "usage_metadata": {
                    "environment": env,
                    "test_type": "environment-validation"
                }
            }
        )

        print(f"  ✓ {env.capitalize()} test complete ({response.usage.total_tokens} tokens)")


def main():
    """Run all tracing examples."""
    print("\n" + "="*60)
    print("Distributed Tracing Examples for Perplexity Middleware")
    print("="*60)

    # Run examples
    example_simple_trace()
    example_distributed_trace()
    example_retry_tracking()
    example_multi_environment()

    print("\n" + "="*60)
    print("✅ All tracing examples completed!")
    print("\nTrace Fields Used:")
    print("  • REVENIUM_ENVIRONMENT - Deployment environment")
    print("  • REVENIUM_REGION - Cloud region")
    print("  • REVENIUM_TRACE_TYPE - Categorical trace identifier")
    print("  • REVENIUM_TRACE_NAME - Human-readable trace label")
    print("  • REVENIUM_PARENT_TRANSACTION_ID - Parent transaction for distributed tracing")
    print("  • REVENIUM_TRANSACTION_NAME - Operation name")
    print("  • REVENIUM_RETRY_NUMBER - Retry attempt number")
    print("\nCheck your Revenium dashboard to visualize the traces!")
    print("="*60)


if __name__ == "__main__":
    main()

