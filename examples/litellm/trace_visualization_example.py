"""
Trace Visualization Example for LiteLLM Middleware

This example demonstrates all 8 trace visualization fields:
1. environment - Deployment environment tracking
2. region - Cloud region identifier
3. credential_alias - API key identification
4. trace_type - Workflow category identifier
5. trace_name - Human-readable trace labels
6. parent_transaction_id - Distributed tracing support
7. transaction_name - Operation-level naming
8. retry_number - Retry attempt tracking

This example demonstrates 5 comprehensive scenarios:
1. Basic trace visualization with environment variables
2. Distributed tracing with parent-child relationships
3. Retry tracking for failed operations
4. Multi-region deployment tracking
5. Parameter-based fields (usage_metadata instead of env vars)
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.litellm.client  # noqa: F401

# Import LiteLLM
import litellm

# Suppress LiteLLM debug output
litellm.suppress_debug_info = True


def example_1_basic_trace_visualization():
    """
    Example 1: Basic Trace Visualization
    
    Demonstrates using environment variables to set trace fields.
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Trace Visualization")
    print("=" * 70)
    
    # Set trace visualization fields via environment variables
    os.environ['REVENIUM_ENVIRONMENT'] = 'production'
    os.environ['REVENIUM_REGION'] = 'us-east-1'
    os.environ['REVENIUM_CREDENTIAL_ALIAS'] = 'openai-prod-key'
    os.environ['REVENIUM_TRACE_TYPE'] = 'customer-support'
    os.environ['REVENIUM_TRACE_NAME'] = 'Customer Support Chat'
    os.environ['REVENIUM_TRANSACTION_NAME'] = 'Answer Question'
    
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is your refund policy?"}],
            usage_metadata={
                "organization_name": "acme-corp",
                "product_name": "support-bot",
                "trace_id": f"support-{int(time.time() * 1000)}",
            }
        )
        
        print(f"\n✅ Response: {response.choices[0].message.content[:100]}...")
        print(f"Trace fields captured: environment, region, credential_alias, trace_type, trace_name, transaction_name")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENAI_API_KEY in your .env file")
        print("2. Set REVENIUM_METERING_API_KEY in your .env file")
        print("3. Installed the package: pip install -e .")


def example_2_distributed_tracing():
    """
    Example 2: Distributed Tracing
    
    Demonstrates parent-child transaction relationships for distributed tracing.
    """
    print("\n" + "=" * 70)
    print("Example 2: Distributed Tracing")
    print("=" * 70)
    
    # Parent transaction
    parent_txn_id = f"parent-{int(time.time() * 1000)}"
    
    # Parent call
    print("\n🔵 Parent Transaction: Extract Key Points")
    try:
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Extract 3 key points from: AI is transforming industries."}],
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
        
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Summarize the key points in one sentence."}],
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
        
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Generate 3 tags for this content."}],
            usage_metadata={
                "organization_name": "acme-corp",
                "product_name": "doc-analyzer",
                "trace_id": f"child2-{int(time.time() * 1000)}",
            }
        )
        
        print("Child 2 completed")
        print(f"\n✅ Workflow complete! Parent: {parent_txn_id}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_3_retry_tracking():
    """
    Example 3: Retry Tracking

    Demonstrates tracking retry attempts with retry_number field.
    """
    print("\n" + "=" * 70)
    print("Example 3: Retry Tracking")
    print("=" * 70)

    # Simulate retries
    for retry_num in range(3):
        os.environ['REVENIUM_RETRY_NUMBER'] = str(retry_num)
        print(f"\n🔄 Attempt {retry_num + 1}/3 (retry_number={retry_num})")

        try:
            litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Fetch user data for user ID 12345"}],
                usage_metadata={
                    "organization_name": "acme-corp",
                    "product_name": "user-service",
                    "trace_id": f"retry-{int(time.time() * 1000)}",
                }
            )

            print(f"Attempt {retry_num + 1} completed")
            time.sleep(0.5)  # Small delay between retries

        except Exception as e:
            print(f"Attempt {retry_num + 1} failed: {e}")

    print("\n✅ All retry attempts tracked with retry_number field!")


def example_4_multi_region():
    """
    Example 4: Multi-Region Deployment

    Demonstrates tracking requests across different cloud regions.
    """
    print("\n" + "=" * 70)
    print("Example 4: Multi-Region Deployment")
    print("=" * 70)

    # Simulate requests from different regions
    regions = ['us-east-1', 'eu-west-1', 'ap-southeast-1']

    for region in regions:
        os.environ['REVENIUM_REGION'] = region
        print(f"\n🌍 Processing in region: {region}")

        try:
            litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Generate a welcome message for users in {region}"}],
                usage_metadata={
                    "organization_name": "acme-corp",
                    "product_name": "content-gen",
                    "trace_id": f"region-{region}-{int(time.time() * 1000)}",
                }
            )

            print(f"Region {region} completed")

        except Exception as e:
            print(f"Region {region} failed: {e}")

    print("\n✅ Multi-region tracking complete!")


def example_5_parameter_based_fields():
    """
    Example 5: Parameter-Based Fields

    Demonstrates passing trace fields via usage_metadata instead of environment variables.
    """
    print("\n" + "=" * 70)
    print("Example 5: Parameter-Based Fields")
    print("=" * 70)

    # Pass all trace fields via usage_metadata
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What are the benefits of cloud computing?"}],
            usage_metadata={
                "organization_name": "acme-corp",
                "product_name": "knowledge-base",
                "trace_id": f"param-{int(time.time() * 1000)}",
                # Trace visualization fields via parameters
                "environment": "staging",
                "region": "us-west-2",
                "credential_alias": "staging-openai-key",
                "trace_type": "knowledge-retrieval",
                "trace_name": "Cloud Computing FAQ",
                "transaction_name": "Answer Question",
            }
        )

        print(f"\n✅ Response: {response.choices[0].message.content[:100]}...")
        print(f"All trace fields passed via usage_metadata parameter!")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Run all trace visualization examples."""
    try:
        print("\n" + "=" * 70)
        print("🚀 Revenium Trace Visualization Examples - LiteLLM Middleware")
        print("=" * 70)

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
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

