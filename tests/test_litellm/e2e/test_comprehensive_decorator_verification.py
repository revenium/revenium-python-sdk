"""
Comprehensive decorator verification with REAL Revenium API verification.

This test:
1. Sends REAL transactions (NO MOCKS) using each decorator
2. Captures transaction IDs
3. Waits for Revenium to record them
4. Prints transaction IDs for manual MCP verification

SUCCESS CRITERIA:
- Zero 4xx/5xx errors
- All transactions must be verifiable in Revenium
- All expected metadata fields must be present

Run this test, then use the MCP tool to verify each transaction.
"""

import os
import sys
import time
import uuid
from pathlib import Path

# Load .env from repository root
repo_root = Path(__file__).parent.parent
env_file = repo_root / ".env"

if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print(f"✅ Loaded environment from {env_file}")
else:
    print(f"⚠️  No .env file found at {env_file}")

# Verify required environment variables
required_vars = ["OPENAI_API_KEY", "REVENIUM_METERING_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"❌ ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print(f"   Please create a .env file at {env_file} with:")
    for var in missing_vars:
        print(f"   {var}=your_key_here")
    sys.exit(1)

print("✅ All required environment variables present\n")

# Import middleware FIRST
import revenium_middleware.litellm.client.middleware

try:
    import litellm
    print("✅ LiteLLM available\n")
except ImportError:
    print("❌ LiteLLM not installed")
    sys.exit(1)

from revenium_middleware.litellm.client import (
    track_agent,
    track_task,
    track_trace,
    track_organization,
    track_subscription,
    track_product,
    track_subscriber,
    track_quality
)

# Test results
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "transactions": []
}


def run_test(test_name: str, decorator_func, expected_fields: dict):
    """
    Run a single decorator test with REAL API call.
    
    Args:
        test_name: Name of the test
        decorator_func: The decorated function to call
        expected_fields: Dictionary of expected metadata fields
    
    Returns:
        True if test passed, False otherwise
    """
    test_results["total"] += 1
    print(f"\n{'='*70}")
    print(f"TEST {test_results['total']}: {test_name}")
    print(f"{'='*70}")
    
    try:
        # Call the decorated function (makes REAL API call)
        response = decorator_func()
        
        if not response:
            print(f"❌ FAILED: No response from LiteLLM")
            test_results["failed"] += 1
            return False
        
        # Extract transaction ID
        transaction_id = response.get("id", "UNKNOWN")
        print(f"✅ Transaction sent successfully")
        print(f"   Transaction ID: {transaction_id}")
        print(f"   Model: {response.get('model', 'UNKNOWN')}")
        
        # Store for verification
        test_results["transactions"].append({
            "test_name": test_name,
            "transaction_id": transaction_id,
            "expected_fields": expected_fields
        })
        
        print(f"\n   Expected metadata fields:")
        for field, value in expected_fields.items():
            print(f"     - {field}: {value}")
        
        test_results["passed"] += 1
        return True
        
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")
        test_results["failed"] += 1
        return False


# Test 1: Agent Decorator
def test_agent():
    agent_name = f"TestAgent-{uuid.uuid4().hex[:8]}"
    
    @track_agent(agent_name)
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test agent"}]
        )
    
    return run_test(
        "Agent Decorator",
        make_call,
        {"agent": agent_name}
    )


# Test 2: Task Decorator
def test_task():
    task_type = f"test_task_{uuid.uuid4().hex[:8]}"
    
    @track_task(task_type)
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test task"}]
        )
    
    return run_test(
        "Task Decorator",
        make_call,
        {"taskType": task_type}
    )


# Test 3: Trace Decorator
def test_trace():
    trace_id = f"trace-{uuid.uuid4()}"
    
    @track_trace(trace_id)
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test trace"}]
        )
    
    return run_test(
        "Trace Decorator",
        make_call,
        {"traceId": trace_id}
    )


# Test 4: Organization Decorator
def test_organization():
    org_id = f"TestOrg-{uuid.uuid4().hex[:8]}"
    
    @track_organization(org_id)
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test organization"}]
        )
    
    return run_test(
        "Organization Decorator",
        make_call,
        {"organization.label": org_id}
    )


# Test 5: Subscription Decorator
def test_subscription():
    sub_id = f"sub-{uuid.uuid4().hex[:12]}"
    
    @track_subscription(sub_id)
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test subscription"}]
        )
    
    return run_test(
        "Subscription Decorator",
        make_call,
        {"subscriptionId": sub_id}
    )


# Test 6: Product Decorator
def test_product():
    product_id = f"TestProduct-{uuid.uuid4().hex[:8]}"
    
    @track_product(product_id)
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test product"}]
        )
    
    return run_test(
        "Product Decorator",
        make_call,
        {"product.label": product_id}
    )


# Test 7: Subscriber Decorator
def test_subscriber():
    subscriber_id = f"user-{uuid.uuid4().hex[:8]}"
    subscriber_email = f"test-{uuid.uuid4().hex[:6]}@example.com"
    
    @track_subscriber(
        subscriber_id=subscriber_id,
        subscriber_email=subscriber_email
    )
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test subscriber"}]
        )
    
    return run_test(
        "Subscriber Decorator",
        make_call,
        {
            "subscriberId": subscriber_id,
            "subscriberEmail": subscriber_email
        }
    )


# Test 8: Quality Decorator
def test_quality():
    quality_score = 0.95
    
    @track_quality(quality_score)
    def make_call():
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test quality"}]
        )
    
    return run_test(
        "Quality Decorator",
        make_call,
        {"responseQualityScore": quality_score}
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE DECORATOR VERIFICATION TEST")
    print("="*70)
    print("\nThis test sends REAL transactions to verify decorator functionality.")
    print("Each transaction will be sent to OpenAI and Revenium.\n")
    
    # Run all tests
    test_agent()
    test_task()
    test_trace()
    test_organization()
    test_subscription()
    test_product()
    test_subscriber()
    test_quality()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")
    
    # Print verification instructions
    print("\n" + "="*70)
    print("REVENIUM VERIFICATION REQUIRED")
    print("="*70)
    print("\nUse the MCP tool to verify each transaction:")
    print("\nmanage_metering_revenium(")
    print("    action='lookup_recent_transactions',")
    print("    page=0,")
    print("    page_size=10,")
    print("    return_transaction_data='full'")
    print(")\n")
    
    print("Verify these transactions:")
    for i, tx in enumerate(test_results["transactions"], 1):
        print(f"\n{i}. {tx['test_name']}")
        print(f"   Transaction ID: {tx['transaction_id']}")
        print(f"   Expected fields:")
        for field, value in tx["expected_fields"].items():
            print(f"     - {field}: {value}")
    
    if test_results["failed"] == 0:
        print("\n✅ ALL TESTS PASSED - Ready for Revenium verification")
        sys.exit(0)
    else:
        print(f"\n❌ {test_results['failed']} TEST(S) FAILED")
        sys.exit(1)

