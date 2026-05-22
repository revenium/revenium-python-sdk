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
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.openai.middleware  # noqa: F401


def example_1_basic_trace_visualization():
    """Example 1: Basic trace visualization with environment variables."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Trace Visualization")
    print("=" * 70)
    
    # Set trace visualization environment variables
    os.environ['REVENIUM_ENVIRONMENT'] = 'production'
    os.environ['REVENIUM_REGION'] = 'us-east-1'
    os.environ['REVENIUM_CREDENTIAL_ALIAS'] = 'openai-prod-key'
    os.environ['REVENIUM_TRACE_TYPE'] = 'customer-support'
    os.environ['REVENIUM_TRACE_NAME'] = 'Customer Support Chat Session'

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is your refund policy?"}
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "support-bot",
            "trace_id": f"support-{int(time.time() * 1000)}",
        }
    )

    print(f"Response: {response.choices[0].message.content[:100]}...")
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

    client = OpenAI()

    # Parent call
    print("\n🔵 Parent Transaction: Extract Key Points")
    parent_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Extract 3 key points from: AI is transforming industries."
            }
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "doc-analyzer",
            "trace_id": parent_txn_id,
        }
    )

    print(f"Parent completed: {parent_response.usage.total_tokens} tokens")

    # Child transaction 1
    print("\nChild Transaction 1: Summarize Points")
    os.environ['REVENIUM_PARENT_TRANSACTION_ID'] = parent_txn_id
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Summarize Points'

    child1_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Summarize these points in one sentence."
            }
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "doc-analyzer",
            "trace_id": f"child1-{int(time.time() * 1000)}",
        }
    )

    print(f"Child 1 completed: {child1_response.usage.total_tokens} tokens")

    # Child transaction 2
    print("\nChild Transaction 2: Generate Tags")
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Generate Tags'

    child2_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Generate 3 tags for this content."}
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "doc-analyzer",
            "trace_id": f"child2-{int(time.time() * 1000)}",
        }
    )

    print(f"Child 2 completed: {child2_response.usage.total_tokens} tokens")
    print("\nWorkflow completed with 1 parent + 2 child transactions")

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

    client = OpenAI()

    # Simulate retry attempts
    for retry_num in range(3):
        os.environ['REVENIUM_RETRY_NUMBER'] = str(retry_num)

        print(f"\nAttempt {retry_num + 1} (retry_number={retry_num})")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Attempt {retry_num + 1}: Say 'Success'"
                }
            ],
            usage_metadata={
                "organizationName": "acme-corp",
                "productName": "api-gateway",
                "trace_id": f"retry-demo-{int(time.time() * 1000)}",
            },
            max_tokens=10
        )

        print(f"Response: {response.choices[0].message.content}")

        # Simulate success on attempt 3
        if retry_num == 2:
            print(f"\nSuccess on attempt {retry_num + 1}!")
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

    client = OpenAI()

    for region in regions:
        os.environ['REVENIUM_REGION'] = region
        os.environ['REVENIUM_CREDENTIAL_ALIAS'] = f'openai-{region}-key'

        print(f"\nProcessing in region: {region}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Hello from {region}!"}
            ],
            usage_metadata={
                "organizationName": "acme-corp",
                "productName": "global-app",
                "trace_id": f"region-{region}-{int(time.time() * 1000)}",
            },
            max_tokens=10
        )

        print(f"Response: {response.choices[0].message.content}")
        print(f"Tracked to region: {region}")


def example_5_custom_trace_categorization():
    """Example 5: Custom trace categorization for different workflows."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Trace Categorization")
    print("=" * 70)

    workflows = [
        {
            'type': 'data-extraction',
            'name': 'Invoice Data Extraction',
            'task': 'Extract invoice details',
            'prompt': 'Extract invoice number and total from this text.'
        },
        {
            'type': 'content-generation',
            'name': 'Marketing Copy Generation',
            'task': 'Generate marketing copy',
            'prompt': 'Write a catchy tagline for a coffee shop.'
        },
        {
            'type': 'code-review',
            'name': 'Code Quality Analysis',
            'task': 'Review code quality',
            'prompt': 'Review this Python function for best practices.'
        }
    ]

    client = OpenAI()

    for workflow in workflows:
        os.environ['REVENIUM_TRACE_TYPE'] = workflow['type']
        os.environ['REVENIUM_TRACE_NAME'] = workflow['name']
        os.environ['REVENIUM_TRANSACTION_NAME'] = workflow['task']

        print(f"\nWorkflow: {workflow['name']}")
        print(f"Type: {workflow['type']}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": workflow['prompt']}
            ],
            usage_metadata={
                "organizationName": "acme-corp",
                "productName": "ai-workflows",
                "trace_id": f"{workflow['type']}-{int(time.time() * 1000)}",
                "task_type": workflow['type'],
            },
            max_tokens=30
        )

        print(f"Completed: {response.usage.total_tokens} tokens")


def main():
    """Run all trace visualization examples."""

    print("\n" + "=" * 70)
    print("REVENIUM TRACE VISUALIZATION EXAMPLES")
    print("=" * 70)

    # Verify API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("\nError: OPENAI_API_KEY not set")
        print("   Add it to your .env file: OPENAI_API_KEY=\"sk-...\"")
        return

    if not os.getenv('REVENIUM_METERING_API_KEY'):
        print("\nError: REVENIUM_METERING_API_KEY not set")
        print("   Add it to your .env file:")
        print("   REVENIUM_METERING_API_KEY=\"hak_...\"")
        return

    print("\nAPI keys configured")
    print("\nRunning examples...")

    try:
        example_1_basic_trace_visualization()
        example_2_distributed_tracing()
        example_3_retry_tracking()
        example_4_multi_region_deployment()
        example_5_custom_trace_categorization()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print("\nCheck your Revenium dashboard to see the trace")
        print("   visualization data:")
        print("   https://app.revenium.ai\n")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
