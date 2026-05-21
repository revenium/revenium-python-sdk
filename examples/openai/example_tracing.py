"""
Example demonstrating trace visualization features.

This example shows how to use tracing fields for:
- Retry tracking with retry_number
- Distributed tracing with parent_transaction_id
- Dynamic trace names per session
- Environment and region tracking

Prerequisites:
    Create a .env file in your project directory with:
        OPENAI_API_KEY="your-openai-key"
        REVENIUM_METERING_API_KEY="your-revenium-key"
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import time

# Load environment variables from .env file
load_dotenv()

# Import middleware to enable usage_metadata support
import revenium_middleware.openai.middleware

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def example_1_retry_tracking():
    """
    Example 1: Retry Tracking
    
    Demonstrates proper use of retry_number to track retry attempts.
    BEST PRACTICE: Pass retry_number via usage_metadata, not env var.
    """
    print("\n" + "="*70)
    print("Example 1: Retry Tracking with retry_number")
    print("="*70)
    
    max_retries = 3
    prompt = "What is the capital of France?"
    
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 Attempt {attempt + 1}/{max_retries} (retry_number={attempt})")
            
            response = client.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": prompt}],
                usage_metadata={
                    "retry_number": attempt,  # 0 for first attempt, 1+ for retries
                    "trace_id": "retry-example-123",
                    "task_type": "chat-with-retry",
                    "transaction_name": f"Chat Attempt {attempt + 1}"
                },
                max_tokens=50
            )
            
            answer = response.choices[0].message.content
            print(f"✅ Success! Answer: {answer}")
            break
            
        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("🚫 Max retries reached, giving up")
                raise
            print(f"⏳ Retrying in 1 second...")
            time.sleep(1)


def example_2_distributed_tracing():
    """
    Example 2: Distributed Tracing
    
    Demonstrates linking child operations to parent using parent_transaction_id.
    BEST PRACTICE: Pass parent_transaction_id via usage_metadata.
    """
    print("\n" + "="*70)
    print("Example 2: Distributed Tracing with parent_transaction_id")
    print("="*70)
    
    # Parent operation
    print("\n📊 Parent Operation: Document Analysis")
    parent_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{
            "role": "user",
            "content": "Analyze this text: 'AI is transforming software development.'"
        }],
        usage_metadata={
            "trace_id": "distributed-trace-456",
            "trace_type": "document-analysis",
            "transaction_name": "Analyze Document",
            "task_type": "analysis"
        },
        max_tokens=100
    )
    
    parent_txn_id = parent_response.id
    print(f"✅ Parent completed. Transaction ID: {parent_txn_id}")
    print(f"   Analysis: {parent_response.choices[0].message.content[:100]}...")
    
    # Child operation 1: Summarize
    print("\n📝 Child Operation 1: Summarize Findings")
    child1_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{
            "role": "user",
            "content": "Summarize: AI is transforming software development."
        }],
        usage_metadata={
            "trace_id": "distributed-trace-456",
            "parent_transaction_id": parent_txn_id,  # Link to parent
            "trace_type": "document-analysis",
            "transaction_name": "Summarize Results",
            "task_type": "summarization"
        },
        max_tokens=50
    )
    
    print(f"✅ Child 1 completed. Transaction ID: {child1_response.id}")
    print(f"   Summary: {child1_response.choices[0].message.content}")
    
    # Child operation 2: Extract keywords
    print("\n🔑 Child Operation 2: Extract Keywords")
    child2_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{
            "role": "user",
            "content": "Extract 3 keywords from: AI is transforming software development."
        }],
        usage_metadata={
            "trace_id": "distributed-trace-456",
            "parent_transaction_id": parent_txn_id,  # Link to same parent
            "trace_type": "document-analysis",
            "transaction_name": "Extract Keywords",
            "task_type": "extraction"
        },
        max_tokens=30
    )
    
    print(f"✅ Child 2 completed. Transaction ID: {child2_response.id}")
    print(f"   Keywords: {child2_response.choices[0].message.content}")
    
    print(f"\n📈 Trace Summary:")
    print(f"   Parent: {parent_txn_id}")
    print(f"   Child 1: {child1_response.id} → {parent_txn_id}")
    print(f"   Child 2: {child2_response.id} → {parent_txn_id}")


def example_3_dynamic_trace_names():
    """
    Example 3: Dynamic Trace Names

    Demonstrates using dynamic trace names for user sessions.
    BEST PRACTICE: Use usage_metadata for per-session trace names.
    """
    print("\n" + "="*70)
    print("Example 3: Dynamic Trace Names per User Session")
    print("="*70)

    # Simulate different user sessions
    sessions = [
        {"user_id": "user-001", "session_id": "session-abc", "message": "Hello!"},
        {"user_id": "user-002", "session_id": "session-def", "message": "Help me code"},
        {"user_id": "user-001", "session_id": "session-ghi", "message": "Another question"}
    ]

    for session in sessions:
        print(f"\n👤 User: {session['user_id']}, Session: {session['session_id']}")

        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": session['message']}],
            usage_metadata={
                "trace_id": session['session_id'],
                "trace_name": f"User {session['user_id']} - {session['session_id']}",  # Dynamic
                "trace_type": "customer-support",
                "transaction_name": "Chat Response",
                "subscriber": {"id": session['user_id']},
                "task_type": "chat"
            },
            max_tokens=30
        )

        print(f"✅ Response: {response.choices[0].message.content}")


def example_4_environment_and_region():
    """
    Example 4: Environment and Region Tracking

    Demonstrates how environment and region are typically set via env vars
    but can be overridden via usage_metadata.
    """
    print("\n" + "="*70)
    print("Example 4: Environment and Region Tracking")
    print("="*70)

    # These would typically come from environment variables:
    # REVENIUM_ENVIRONMENT=production
    # REVENIUM_REGION=us-east-1

    print("\n🌍 Using environment variables for static deployment info")
    print(f"   Environment: {os.getenv('REVENIUM_ENVIRONMENT', 'not-set')}")
    print(f"   Region: {os.getenv('REVENIUM_REGION', 'not-set')}")

    # Normal call - uses env vars
    response1 = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Test message"}],
        usage_metadata={
            "trace_id": "env-test-1",
            "task_type": "test"
        },
        max_tokens=20
    )
    print(f"\n✅ Call 1: Uses env vars for environment/region")

    # Override for special case (e.g., cross-region call)
    response2 = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Cross-region test"}],
        usage_metadata={
            "trace_id": "env-test-2",
            "environment": "staging",  # Override env var
            "region": "eu-west-1",     # Override env var
            "task_type": "cross-region-test"
        },
        max_tokens=20
    )
    print(f"✅ Call 2: Overrides with environment='staging', region='eu-west-1'")


def example_5_complete_tracing_scenario():
    """
    Example 5: Complete Tracing Scenario

    Demonstrates a realistic scenario combining multiple tracing features.
    """
    print("\n" + "="*70)
    print("Example 5: Complete Tracing Scenario - Multi-Step Workflow")
    print("="*70)

    workflow_id = "workflow-789"
    user_id = "user-premium-001"

    # Step 1: Initial request (with retry logic)
    print(f"\n📋 Step 1: Initial Request")
    for attempt in range(2):
        try:
            step1_response = client.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Explain quantum computing"}],
                usage_metadata={
                    "trace_id": workflow_id,
                    "trace_type": "educational-workflow",
                    "trace_name": f"User {user_id} - Quantum Learning",
                    "transaction_name": "Initial Explanation",
                    "retry_number": attempt,
                    "subscriber": {"id": user_id},
                    "task_type": "education"
                },
                max_tokens=100
            )
            print(f"✅ Step 1 completed (attempt {attempt + 1})")
            break
        except Exception as e:
            if attempt == 1:
                raise
            time.sleep(0.5)

    parent_txn = step1_response.id

    # Step 2: Follow-up question (child of step 1)
    print(f"\n❓ Step 2: Follow-up Question")
    step2_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Give me an example"}],
        usage_metadata={
            "trace_id": workflow_id,
            "parent_transaction_id": parent_txn,
            "trace_type": "educational-workflow",
            "trace_name": f"User {user_id} - Quantum Learning",
            "transaction_name": "Example Request",
            "subscriber": {"id": user_id},
            "task_type": "education"
        },
        max_tokens=80
    )
    print(f"✅ Step 2 completed")

    # Step 3: Summary (child of step 1)
    print(f"\n📝 Step 3: Generate Summary")
    step3_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Summarize quantum computing in one sentence"}],
        usage_metadata={
            "trace_id": workflow_id,
            "parent_transaction_id": parent_txn,
            "trace_type": "educational-workflow",
            "trace_name": f"User {user_id} - Quantum Learning",
            "transaction_name": "Generate Summary",
            "subscriber": {"id": user_id},
            "task_type": "summarization"
        },
        max_tokens=50
    )
    print(f"✅ Step 3 completed")

    print(f"\n📊 Workflow Complete!")
    print(f"   Trace ID: {workflow_id}")
    print(f"   Parent Transaction: {parent_txn}")
    print(f"   Child Transactions: {step2_response.id}, {step3_response.id}")


if __name__ == "__main__":
    print("="*70)
    print("OpenAI Middleware - Trace Visualization Examples")
    print("="*70)

    try:
        example_1_retry_tracking()
        example_2_distributed_tracing()
        example_3_dynamic_trace_names()
        example_4_environment_and_region()
        example_5_complete_tracing_scenario()

        print("\n" + "="*70)
        print("✅ All tracing examples completed!")
        print("="*70)
        print("\n💡 Key Takeaways:")
        print("   1. Use retry_number in usage_metadata to track retry attempts")
        print("   2. Use parent_transaction_id to link child operations to parents")
        print("   3. Use dynamic trace_name for per-session/per-user tracking")
        print("   4. Use env vars for static deployment info (environment, region)")
        print("   5. Combine all features for comprehensive distributed tracing")
        print("\n📊 Check your Revenium dashboard to see the trace visualization!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


