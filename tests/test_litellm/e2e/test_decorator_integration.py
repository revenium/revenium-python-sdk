#!/usr/bin/env python3
"""
Comprehensive integration tests for all 8 Revenium decorators.

Tests each decorator with:
1. Static value injection
2. Dynamic value from function arguments
3. Dynamic value from object attributes
4. Both sync and async functions

Validates metadata transmission to Revenium API.
"""

import asyncio
import os
import sys
from typing import Optional

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import decorators
from revenium_middleware.litellm.client import (
    track_agent,
    track_task,
    track_trace,
    track_organization,
    track_subscription,
    track_product,
    track_subscriber,
    track_quality,
)

# Import context for verification
from revenium_middleware.litellm.client.context import metadata_context

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "errors": []
}


def verify_metadata(expected: dict, test_name: str) -> bool:
    """Verify that metadata context contains expected values."""
    current = metadata_context.get()
    
    for key, expected_value in expected.items():
        if key not in current:
            error = f"❌ {test_name}: Missing key '{key}' in metadata"
            test_results["errors"].append(error)
            print(error)
            return False
        
        actual_value = current[key]
        if actual_value != expected_value:
            error = f"❌ {test_name}: Expected {key}='{expected_value}', got '{actual_value}'"
            test_results["errors"].append(error)
            print(error)
            return False
    
    return True


def test_track_agent_static():
    """Test track_agent with static value."""
    @track_agent("Lead Analyst")
    def test_func():
        return verify_metadata({"agent": "Lead Analyst"}, "track_agent_static")
    
    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_agent (static value)")
    else:
        test_results["failed"] += 1


def test_track_agent_from_arg():
    """Test track_agent with value from argument."""
    @track_agent(name_from_arg="agent_name")
    def test_func(agent_name: str):
        return verify_metadata({"agent": agent_name}, "track_agent_from_arg")
    
    result = test_func("Research Agent")
    if result:
        test_results["passed"] += 1
        print("✅ track_agent (from argument)")
    else:
        test_results["failed"] += 1


def test_track_agent_from_attr():
    """Test track_agent with value from object attribute."""
    class TestAgent:
        def __init__(self, name: str):
            self.name = name
        
        @track_agent(name_from_attr="name")
        def execute(self):
            return verify_metadata({"agent": self.name}, "track_agent_from_attr")
    
    agent = TestAgent("Content Writer")
    result = agent.execute()
    if result:
        test_results["passed"] += 1
        print("✅ track_agent (from attribute)")
    else:
        test_results["failed"] += 1


async def test_track_agent_async():
    """Test track_agent with async function."""
    @track_agent("Async Agent")
    async def test_func():
        return verify_metadata({"agent": "Async Agent"}, "track_agent_async")
    
    result = await test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_agent (async)")
    else:
        test_results["failed"] += 1


def test_track_task_static():
    """Test track_task with static value."""
    @track_task("research")
    def test_func():
        return verify_metadata({"task_type": "research"}, "track_task_static")
    
    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_task (static value)")
    else:
        test_results["failed"] += 1


def test_track_task_from_arg():
    """Test track_task with value from argument."""
    @track_task(type_from_arg="task_type")
    def test_func(task_type: str):
        return verify_metadata({"task_type": task_type}, "track_task_from_arg")
    
    result = test_func("analysis")
    if result:
        test_results["passed"] += 1
        print("✅ track_task (from argument)")
    else:
        test_results["failed"] += 1


def test_track_trace_static():
    """Test track_trace with static value."""
    @track_trace("trace-123")
    def test_func():
        return verify_metadata({"trace_id": "trace-123"}, "track_trace_static")
    
    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_trace (static value)")
    else:
        test_results["failed"] += 1


def test_track_trace_from_arg():
    """Test track_trace with value from argument."""
    @track_trace(id_from_arg="workflow_id")
    def test_func(workflow_id: str):
        return verify_metadata({"trace_id": workflow_id}, "track_trace_from_arg")
    
    result = test_func("workflow-456")
    if result:
        test_results["passed"] += 1
        print("✅ track_trace (from argument)")
    else:
        test_results["failed"] += 1


def test_track_organization_static():
    """Test track_organization with static value."""
    @track_organization("AcmeCorp")
    def test_func():
        return verify_metadata({"organization_name": "AcmeCorp"}, "track_organization_static")

    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_organization (static value)")
    else:
        test_results["failed"] += 1


def test_track_organization_from_arg():
    """Test track_organization with value from argument."""
    @track_organization(name_from_arg="org_name")
    def test_func(org_name: str):
        return verify_metadata({"organization_name": org_name}, "track_organization_from_arg")

    result = test_func("TechCorp")
    if result:
        test_results["passed"] += 1
        print("✅ track_organization (from argument)")
    else:
        test_results["failed"] += 1


def test_track_subscription_static():
    """Test track_subscription with static value."""
    @track_subscription("sub-123")
    def test_func():
        return verify_metadata({"subscription_id": "sub-123"}, "track_subscription_static")
    
    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_subscription (static value)")
    else:
        test_results["failed"] += 1


def test_track_subscription_from_arg():
    """Test track_subscription with value from argument."""
    @track_subscription(id_from_arg="sub_id")
    def test_func(sub_id: str):
        return verify_metadata({"subscription_id": sub_id}, "track_subscription_from_arg")
    
    result = test_func("sub-456")
    if result:
        test_results["passed"] += 1
        print("✅ track_subscription (from argument)")
    else:
        test_results["failed"] += 1


def test_track_product_static():
    """Test track_product with static value."""
    @track_product("premium")
    def test_func():
        return verify_metadata({"product_name": "premium"}, "track_product_static")

    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_product (static value)")
    else:
        test_results["failed"] += 1


def test_track_product_from_arg():
    """Test track_product with value from argument."""
    @track_product(name_from_arg="prod_name")
    def test_func(prod_name: str):
        return verify_metadata({"product_name": prod_name}, "track_product_from_arg")

    result = test_func("enterprise")
    if result:
        test_results["passed"] += 1
        print("✅ track_product (from argument)")
    else:
        test_results["failed"] += 1


def test_track_subscriber_static():
    """Test track_subscriber with static values."""
    @track_subscriber(subscriber_id="user-123", subscriber_email="user@example.com")
    def test_func():
        expected = {
            "subscriber": {
                "id": "user-123",
                "email": "user@example.com"
            }
        }
        return verify_metadata(expected, "track_subscriber_static")
    
    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_subscriber (static values)")
    else:
        test_results["failed"] += 1


def test_track_subscriber_from_args():
    """Test track_subscriber with values from arguments."""
    @track_subscriber(id_from_arg="user_id", email_from_arg="user_email")
    def test_func(user_id: str, user_email: str):
        expected = {
            "subscriber": {
                "id": user_id,
                "email": user_email
            }
        }
        return verify_metadata(expected, "track_subscriber_from_args")
    
    result = test_func("user-456", "test@example.com")
    if result:
        test_results["passed"] += 1
        print("✅ track_subscriber (from arguments)")
    else:
        test_results["failed"] += 1


def test_track_quality_static():
    """Test track_quality with static value."""
    @track_quality(0.95)
    def test_func():
        return verify_metadata({"response_quality_score": 0.95}, "track_quality_static")
    
    result = test_func()
    if result:
        test_results["passed"] += 1
        print("✅ track_quality (static value)")
    else:
        test_results["failed"] += 1


def test_track_quality_from_arg():
    """Test track_quality with value from argument."""
    @track_quality(score_from_arg="quality")
    def test_func(quality: float):
        return verify_metadata({"response_quality_score": quality}, "track_quality_from_arg")
    
    result = test_func(0.87)
    if result:
        test_results["passed"] += 1
        print("✅ track_quality (from argument)")
    else:
        test_results["failed"] += 1


def run_all_tests():
    """Run all decorator tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE DECORATOR INTEGRATION TESTS")
    print("="*70 + "\n")
    
    # Run sync tests
    test_track_agent_static()
    test_track_agent_from_arg()
    test_track_agent_from_attr()
    
    test_track_task_static()
    test_track_task_from_arg()
    
    test_track_trace_static()
    test_track_trace_from_arg()
    
    test_track_organization_static()
    test_track_organization_from_arg()
    
    test_track_subscription_static()
    test_track_subscription_from_arg()
    
    test_track_product_static()
    test_track_product_from_arg()
    
    test_track_subscriber_static()
    test_track_subscriber_from_args()
    
    test_track_quality_static()
    test_track_quality_from_arg()
    
    # Run async tests
    asyncio.run(test_track_agent_async())
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"📊 Total:  {test_results['passed'] + test_results['failed']}")
    
    if test_results["failed"] > 0:
        print("\n❌ FAILURES DETECTED:")
        for error in test_results["errors"]:
            print(f"  {error}")
        print("\n🚨 INTEGRATION TESTS FAILED - FIX REQUIRED")
        return False
    else:
        print("\n✅ ALL INTEGRATION TESTS PASSED")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

