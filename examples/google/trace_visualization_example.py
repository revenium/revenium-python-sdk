#!/usr/bin/env python3
"""
Trace Visualization Example for Google AI Middleware

Demonstrates how to use Revenium's trace visualization features for
distributed tracing, retry tracking, and custom trace categorization.

Features demonstrated:
1. Basic trace visualization with environment variables
2. Distributed tracing with parent-child relationships
3. Retry tracking for failed operations
4. Multi-region deployment tracking
5. Parameter-based fields (usage_metadata instead of env vars)

This example uses Google AI SDK (Gemini Developer API).
For Vertex AI, the pattern is identical - just use vertexai instead of google.genai.
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.google  # noqa: F401

# Import Google AI SDK
from google import genai

# Create client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def example_1_basic_trace_visualization():
    """Example 1: Basic trace visualization with environment variables."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Trace Visualization")
    print("=" * 70)

    # Set trace visualization environment variables
    os.environ['REVENIUM_ENVIRONMENT'] = 'production'
    os.environ['REVENIUM_REGION'] = 'us-east-1'
    os.environ['REVENIUM_CREDENTIAL_ALIAS'] = 'google-prod-key'
    os.environ['REVENIUM_TRACE_TYPE'] = 'customer-support'
    os.environ['REVENIUM_TRACE_NAME'] = 'Customer Support Chat Session'

    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents="What is your refund policy?",
        usage_metadata={
            "organization_name": "acme-corp",
            "product_name": "support-bot",
            "trace_id": f"support-{int(time.time() * 1000)}",
        }
    )

    print(f"Response: {response.text[:100]}...")
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
    client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents="Extract 3 key points from: AI is transforming industries.",
        usage_metadata={
            "organization_name": "acme-corp",
            "product_name": "doc-analyzer",
            "trace_id": parent_txn_id,
        }
    )

    print("Parent completed")

    # Child transaction 1
    print("\n🟢 Child Transaction 1: Summarize Points")
    os.environ['REVENIUM_PARENT_TRANSACTION_ID'] = parent_txn_id
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Summarize Points'

    client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents="Summarize the key points in one sentence.",
        usage_metadata={
            "organization_name": "acme-corp",
            "product_name": "doc-analyzer",
            "trace_id": f"child1-{int(time.time() * 1000)}",
        }
    )

    print("Child 1 completed")

    # Child transaction 2
    print("\n🟢 Child Transaction 2: Generate Tags")
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Generate Tags'

    client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents="Generate 3 tags for this content.",
        usage_metadata={
            "organization_name": "acme-corp",
            "product_name": "doc-analyzer",
            "trace_id": f"child2-{int(time.time() * 1000)}",
        }
    )

    print("Child 2 completed")
    print(f"\n✅ Workflow complete! Parent: {parent_txn_id}")


def example_3_retry_tracking():
    """Example 3: Retry tracking for failed operations."""
    print("\n" + "=" * 70)
    print("Example 3: Retry Tracking")
    print("=" * 70)

    os.environ['REVENIUM_TRACE_TYPE'] = 'api-integration'
    os.environ['REVENIUM_TRACE_NAME'] = 'External API Call with Retries'
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Fetch User Data'

    # Simulate retries
    for retry_num in range(3):
        os.environ['REVENIUM_RETRY_NUMBER'] = str(retry_num)
        print(f"\n🔄 Attempt {retry_num + 1}/3 (retry_number={retry_num})")

        client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents="Fetch user data for user ID 12345",
            usage_metadata={
                "organization_name": "acme-corp",
                "product_name": "user-service",
                "trace_id": f"retry-{int(time.time() * 1000)}",
            }
        )

        print(f"Attempt {retry_num + 1} completed")
        time.sleep(0.5)  # Small delay between retries

    print("\n✅ All retry attempts tracked with retry_number field!")


def example_4_multi_region():
    """Example 4: Multi-region deployment tracking."""
    print("\n" + "=" * 70)
    print("Example 4: Multi-Region Deployment Tracking")
    print("=" * 70)

    os.environ['REVENIUM_TRACE_TYPE'] = 'content-generation'
    os.environ['REVENIUM_TRACE_NAME'] = 'Multi-Region Content Generation'

    # Simulate requests from different regions
    regions = ['us-east-1', 'eu-west-1', 'ap-southeast-1']

    for region in regions:
        os.environ['REVENIUM_REGION'] = region
        print(f"\n🌍 Processing in region: {region}")

        client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=f"Generate a welcome message for users in {region}",
            usage_metadata={
                "organization_name": "acme-corp",
                "product_name": "content-gen",
                "trace_id": f"region-{region}-{int(time.time() * 1000)}",
            }
        )

        print(f"Region {region} completed")

    print("\n✅ Multi-region tracking complete!")


def example_5_parameter_based_fields():
    """Example 5: Using usage_metadata instead of environment variables."""
    print("\n" + "=" * 70)
    print("Example 5: Parameter-Based Trace Fields")
    print("=" * 70)

    # Clear environment variables to demonstrate parameter-based approach
    for key in ['REVENIUM_ENVIRONMENT', 'REVENIUM_REGION',
                'REVENIUM_TRACE_TYPE', 'REVENIUM_TRACE_NAME',
                'REVENIUM_CREDENTIAL_ALIAS']:
        os.environ.pop(key, None)

    # Pass all trace fields via usage_metadata
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents="What are the benefits of cloud computing?",
        usage_metadata={
            "organization_name": "acme-corp",
            "product_name": "knowledge-base",
            "trace_id": f"param-{int(time.time() * 1000)}",
            # Trace visualization fields via parameters
            "environment": "staging",
            "region": "us-west-2",
            "credential_alias": "staging-google-key",
            "trace_type": "knowledge-retrieval",
            "trace_name": "Cloud Computing FAQ",
            "transaction_name": "Answer Question",
        }
    )

    print(f"Response: {response.text[:100]}...")
    print("\n✅ All trace fields passed via usage_metadata!")
    print("   - environment: staging")
    print("   - region: us-west-2")
    print("   - trace_type: knowledge-retrieval")
    print("   - trace_name: Cloud Computing FAQ")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("🚀 Revenium Trace Visualization Examples - Google AI Middleware")
    print("=" * 70)

    try:
        example_1_basic_trace_visualization()
        example_2_distributed_tracing()
        example_3_retry_tracking()
        example_4_multi_region()
        example_5_parameter_based_fields()

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        print("\nCheck your Revenium dashboard to see the trace visualization data.")
        print("Visit: https://app.revenium.io")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set GOOGLE_API_KEY in your .env file")
        print("2. Set REVENIUM_METERING_API_KEY in your .env file")
        print("3. Installed the package: pip install -e .")


if __name__ == "__main__":
    main()

