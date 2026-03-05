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
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.anthropic.middleware  # noqa: F401


def example_1_basic_trace_visualization():
    """Example 1: Basic trace visualization with environment variables."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Trace Visualization")
    print("=" * 70)

    # Set trace visualization environment variables
    os.environ['REVENIUM_ENVIRONMENT'] = 'production'
    os.environ['REVENIUM_REGION'] = 'us-east-1'
    os.environ['REVENIUM_CREDENTIAL_ALIAS'] = 'anthropic-prod-key'
    os.environ['REVENIUM_TRACE_TYPE'] = 'customer-support'
    os.environ['REVENIUM_TRACE_NAME'] = 'Customer Support Chat Session'

    client = Anthropic()

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "What is your refund policy?"}
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "support-bot",
            "trace_id": f"support-{int(time.time() * 1000)}",
        }
    )

    print(f"Response: {response.content[0].text[:100]}...")
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

    client = Anthropic()

    # Parent call
    print("\n🔵 Parent Transaction: Extract Key Points")
    parent_response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
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

    print(f"Parent completed: {parent_response.usage.input_tokens + parent_response.usage.output_tokens} tokens")

    # Child transaction 1
    print("\n  🟢 Child Transaction 1: Summarize Points")
    os.environ['REVENIUM_PARENT_TRANSACTION_ID'] = parent_txn_id
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Summarize Points'

    child1_response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[
            {"role": "user", "content": "Summarize in one sentence: AI transforms industries"}
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "doc-analyzer",
            "trace_id": f"child1-{int(time.time() * 1000)}",
        }
    )

    print(f"  Child 1 completed: {child1_response.usage.input_tokens + child1_response.usage.output_tokens} tokens")

    # Child transaction 2
    print("\n  🟢 Child Transaction 2: Generate Tags")
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Generate Tags'

    child2_response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[
            {"role": "user", "content": "Generate 3 tags for: AI transformation"}
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "doc-analyzer",
            "trace_id": f"child2-{int(time.time() * 1000)}",
        }
    )

    print(f"  Child 2 completed: {child2_response.usage.input_tokens + child2_response.usage.output_tokens} tokens")
    print("\n✅ Workflow complete! All transactions linked via parent_transaction_id")





def example_3_retry_tracking():
    """Example 3: Retry tracking for failed operations."""
    print("\n" + "=" * 70)
    print("Example 3: Retry Tracking")
    print("=" * 70)

    os.environ['REVENIUM_TRACE_TYPE'] = 'api-integration'
    os.environ['REVENIUM_TRACE_NAME'] = 'External API Call with Retries'
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Fetch User Data'

    client = Anthropic()

    # Simulate retries
    max_retries = 3
    for retry in range(max_retries):
        os.environ['REVENIUM_RETRY_NUMBER'] = str(retry)

        print(f"\n🔄 Attempt {retry + 1}/{max_retries} (retry_number={retry})")

        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[
                    {"role": "user", "content": "Say 'Success!'"}
                ],
                usage_metadata={
                    "organizationName": "acme-corp",
                    "productName": "api-gateway",
                    "trace_id": f"retry-{int(time.time() * 1000)}",
                }
            )

            print(f"✅ Success on attempt {retry + 1}")
            print(f"Response: {response.content[0].text}")
            break
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            if retry == max_retries - 1:
                print("Max retries reached")
            time.sleep(0.5)


def example_4_custom_categorization():
    """Example 4: Custom trace categorization and naming."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Trace Categorization")
    print("=" * 70)

    # Different trace types for different use cases
    trace_configs = [
        {
            'type': 'user-onboarding',
            'name': 'Welcome Email Generation',
            'transaction': 'Generate Welcome Message',
            'content': 'Write a welcome message for a new user'
        },
        {
            'type': 'content-moderation',
            'name': 'Comment Review',
            'transaction': 'Check Comment Safety',
            'content': 'Is this comment appropriate: "Great product!"'
        },
        {
            'type': 'data-enrichment',
            'name': 'Profile Enhancement',
            'transaction': 'Extract Skills',
            'content': 'Extract skills from: Python developer with 5 years experience'
        }
    ]

    client = Anthropic()

    for config in trace_configs:
        print(f"\n📊 Trace Type: {config['type']}")
        print(f"   Trace Name: {config['name']}")

        os.environ['REVENIUM_TRACE_TYPE'] = config['type']
        os.environ['REVENIUM_TRACE_NAME'] = config['name']
        os.environ['REVENIUM_TRANSACTION_NAME'] = config['transaction']

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[
                {"role": "user", "content": config['content']}
            ],
            usage_metadata={
                "organizationName": "acme-corp",
                "productName": "automation-suite",
                "trace_id": f"{config['type']}-{int(time.time() * 1000)}",
            }
        )

        print(f"   ✅ Completed: {response.usage.input_tokens + response.usage.output_tokens} tokens")


def example_5_usage_metadata_override():
    """Example 5: Override environment variables with usage_metadata."""
    print("\n" + "=" * 70)
    print("Example 5: Usage Metadata Override")
    print("=" * 70)

    # Set default environment variables
    os.environ['REVENIUM_ENVIRONMENT'] = 'staging'
    os.environ['REVENIUM_REGION'] = 'us-west-2'
    os.environ['REVENIUM_TRACE_TYPE'] = 'default-workflow'

    client = Anthropic()

    # Override with usage_metadata (takes precedence)
    print("\n🔧 Environment variables set to:")
    print("   REVENIUM_ENVIRONMENT=staging")
    print("   REVENIUM_REGION=us-west-2")
    print("   REVENIUM_TRACE_TYPE=default-workflow")

    print("\n🔧 Overriding with usage_metadata:")
    print("   environment=production")
    print("   region=eu-west-1")
    print("   traceType=priority-workflow")

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        usage_metadata={
            "organizationName": "acme-corp",
            "productName": "override-test",
            "trace_id": f"override-{int(time.time() * 1000)}",
            # These override environment variables
            "environment": "production",
            "region": "eu-west-1",
            "traceType": "priority-workflow",
            "traceName": "High Priority Request",
        }
    )

    print(f"\n✅ Request sent with overridden values")
    print(f"   Actual environment: production")
    print(f"   Actual region: eu-west-1")
    print(f"   Actual trace_type: priority-workflow")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Revenium Trace Visualization Examples for Anthropic")
    print("=" * 70)

    try:
        example_1_basic_trace_visualization()
        example_2_distributed_tracing()
        example_3_retry_tracking()
        example_4_custom_categorization()
        example_5_usage_metadata_override()

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        print("\nCheck your Revenium dashboard to see the trace visualization data.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Set ANTHROPIC_API_KEY in your .env file")
        print("2. Set REVENIUM_API_KEY in your .env file")
        print("3. Installed required packages: pip install anthropic python-dotenv")
