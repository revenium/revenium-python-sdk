#!/usr/bin/env python3
"""
Trace Visualization Example

Demonstrates how to use Revenium's trace visualization features for
distributed tracing, retry tracking, and custom trace categorization.

Features demonstrated:
1. Basic trace visualization with environment variables
2. Distributed tracing with parent-child relationships
3. Retry tracking for failed operations
4. Custom trace categorization and naming
5. Region and credential tracking
"""

import os
import time
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.ollama.middleware  # noqa: F401


def example_1_basic_trace_visualization():
    """Example 1: Basic trace visualization with environment variables."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Trace Visualization")
    print("=" * 70)

    # Set trace visualization environment variables
    os.environ['REVENIUM_ENVIRONMENT'] = 'production'
    os.environ['REVENIUM_REGION'] = 'us-east-1'
    os.environ['REVENIUM_CREDENTIAL_ALIAS'] = 'ollama-prod-key'
    os.environ['REVENIUM_TRACE_TYPE'] = 'customer-support'
    os.environ['REVENIUM_TRACE_NAME'] = 'Customer Support Chat Session'

    response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[
            {"role": "user", "content": "What is your refund policy?"}
        ],
        usage_metadata={
            "organizationName": "AcmeCorp",
            "productName": "customer-support-bot",
            "trace_id": f"support-{int(time.time() * 1000)}",
        }
    )

    print(f"Response: {response['message']['content'][:100]}...")
    print("Trace Type: customer-support")
    print("Trace Name: Customer Support Chat Session")
    print("Environment: production")
    print("Region: us-east-1")


def example_2_distributed_tracing():
    """Example 2: Distributed tracing with parent-child relationships."""
    print("\n" + "=" * 70)
    print("Example 2: Distributed Tracing (Parent-Child)")
    print("=" * 70)

    # Parent transaction
    parent_txn_id = f"parent-{int(time.time() * 1000)}"

    # Set up parent trace
    os.environ['REVENIUM_TRACE_TYPE'] = 'workflow'
    os.environ['REVENIUM_TRACE_NAME'] = 'Document Analysis Workflow'
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Extract Key Points'

    # Parent call
    print("\n🔵 Parent Transaction: Extract Key Points")
    parent_response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[
            {
                "role": "user",
                "content": "Extract 3 key points from: AI is transforming industries."
            }
        ],
        usage_metadata={
            "organizationName": "AcmeCorp",
            "productName": "document-analyzer",
            "trace_id": parent_txn_id,
        }
    )

    print(f"Parent completed: {parent_response.get('usage', {}).get('total_tokens', 'N/A')} tokens")

    # Child transaction 1
    print("\n🟢 Child Transaction 1: Summarize Points")
    os.environ['REVENIUM_PARENT_TRANSACTION_ID'] = parent_txn_id
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Summarize Points'

    child1_response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[
            {
                "role": "user",
                "content": "Summarize these points in one sentence."
            }
        ],
        usage_metadata={
            "organizationName": "AcmeCorp",
            "productName": "document-analyzer",
            "trace_id": f"child1-{int(time.time() * 1000)}",
        }
    )

    print(f"Child 1 completed: {child1_response.get('usage', {}).get('total_tokens', 'N/A')} tokens")

    # Child transaction 2
    print("\n🟢 Child Transaction 2: Generate Tags")
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Generate Tags'

    child2_response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[
            {"role": "user", "content": "Generate 3 tags for this content."}
        ],
        usage_metadata={
            "organizationName": "AcmeCorp",
            "productName": "document-analyzer",
            "trace_id": f"child2-{int(time.time() * 1000)}",
        }
    )

    print(f"Child 2 completed: {child2_response.get('usage', {}).get('total_tokens', 'N/A')} tokens")
    print("\n✅ Workflow completed with 1 parent + 2 child transactions")

    # Clean up
    os.environ.pop('REVENIUM_PARENT_TRANSACTION_ID', None)


def example_3_retry_tracking():
    """Example 3: Retry tracking for failed operations."""
    print("\n" + "=" * 70)
    print("Example 3: Retry Tracking")
    print("=" * 70)

    os.environ['REVENIUM_TRACE_TYPE'] = 'api-integration'
    os.environ['REVENIUM_TRACE_NAME'] = 'External API Call with Retries'
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Fetch User Data'

    # Simulate retry attempts
    for retry_num in range(3):
        os.environ['REVENIUM_RETRY_NUMBER'] = str(retry_num)

        print(f"\nAttempt {retry_num + 1} (retry_number={retry_num})")

        response = ollama.chat(
            model='qwen2.5:0.5b',
            messages=[
                {
                    "role": "user",
                    "content": f"Attempt {retry_num + 1}: Say 'Success'"
                }
            ],
            usage_metadata={
                "organizationName": "AcmeCorp",
                "productName": "api-gateway",
                "trace_id": f"retry-demo-{int(time.time() * 1000)}",
            }
        )

        print(f"Response: {response['message']['content']}")

        # Simulate success on attempt 3
        if retry_num == 2:
            print(f"\n✅ Success on attempt {retry_num + 1}!")
            break

    # Clean up
    os.environ['REVENIUM_RETRY_NUMBER'] = '0'


def example_4_multi_region_deployment():
    """Example 4: Multi-region deployment tracking."""
    print("\n" + "=" * 70)
    print("Example 4: Multi-Region Deployment")
    print("=" * 70)

    regions = ['us-east-1', 'eu-west-1', 'ap-southeast-1']

    os.environ['REVENIUM_ENVIRONMENT'] = 'production'
    os.environ['REVENIUM_TRACE_TYPE'] = 'global-service'
    os.environ['REVENIUM_TRACE_NAME'] = 'Multi-Region Request'

    for region in regions:
        os.environ['REVENIUM_REGION'] = region
        print(f"\n📍 Processing in region: {region}")

        response = ollama.chat(
            model='qwen2.5:0.5b',
            messages=[
                {"role": "user", "content": f"Hello from {region}!"}
            ],
            usage_metadata={
                "organizationName": "AcmeCorp",
                "productName": "global-app",
                "trace_id": f"region-{region}-{int(time.time() * 1000)}",
            }
        )

        print(f"Response from {region}: {response['message']['content'][:50]}...")

    print("\n✅ Multi-region deployment tracking complete")


def example_5_parameter_based_fields():
    """Example 5: Using usage_metadata parameters instead of environment variables."""
    print("\n" + "=" * 70)
    print("Example 5: Parameter-Based Trace Fields")
    print("=" * 70)

    # Clear any existing trace env vars to demonstrate parameter-based approach
    trace_env_vars = [
        'REVENIUM_ENVIRONMENT', 'REVENIUM_REGION', 'REVENIUM_CREDENTIAL_ALIAS',
        'REVENIUM_TRACE_TYPE', 'REVENIUM_TRACE_NAME', 'REVENIUM_TRANSACTION_NAME',
        'REVENIUM_PARENT_TRANSACTION_ID', 'REVENIUM_RETRY_NUMBER'
    ]
    for var in trace_env_vars:
        os.environ.pop(var, None)

    print("\n✨ All trace fields passed via usage_metadata (no env vars)")

    response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[
            {"role": "user", "content": "Explain trace visualization in one sentence."}
        ],
        usage_metadata={
            "organizationName": "AcmeCorp",
            "productName": "demo-app",
            "trace_id": f"param-demo-{int(time.time() * 1000)}",
            # Trace visualization fields as parameters
            "environment": "staging",
            "region": "us-west-2",
            "credential_alias": "ollama-staging-key",
            "trace_type": "demo",
            "trace_name": "Parameter-Based Demo",
            "transaction_name": "Explain Concept",
        }
    )

    print(f"Response: {response['message']['content'][:100]}...")
    print("\n✅ All trace fields successfully passed via parameters!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("REVENIUM OLLAMA MIDDLEWARE - TRACE VISUALIZATION EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate advanced trace visualization features")
    print("for distributed tracing, retry tracking, and custom categorization.")
    print("\nNote: Make sure Ollama is running locally with qwen2.5:0.5b model available")
    print("=" * 70)

    try:
        example_1_basic_trace_visualization()
        example_2_distributed_tracing()
        example_3_retry_tracking()
        example_4_multi_region_deployment()
        example_5_parameter_based_fields()

        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nCheck your Revenium dashboard to see the trace visualization data.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. qwen2.5:0.5b model is available (ollama pull qwen2.5:0.5b)")
        print("3. REVENIUM_METERING_API_KEY is set in your .env file")


if __name__ == "__main__":
    main()

