#!/usr/bin/env python3
"""
Run all E2E baseline tests to establish current state.

This script runs all E2E tests and generates a baseline report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_test(test_file):
    """Run a single test file and return results."""
    print("\n" + "=" * 80)
    print(f"Running: {test_file.name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=False,
        text=True
    )

    return {
        "name": test_file.stem,
        "file": test_file.name,
        "passed": result.returncode == 0
    }


def main():
    """Run all E2E tests."""
    print("=" * 80)
    print("E2E BASELINE TESTS - CURRENT STATE")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print()
    print("PURPOSE: Establish baseline before making any fixes")
    print("These tests will show what works and what doesn't in the current state.")
    print()

    # Find all test files
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))

    if not test_files:
        print("❌ No test files found!")
        return 1

    print(f"Found {len(test_files)} test files:")
    for tf in test_files:
        print(f"  - {tf.name}")
    print()

    # Run all tests
    results = []
    for test_file in test_files:
        try:
            result = run_test(test_file)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error running {test_file.name}: {e}")
            results.append({
                "name": test_file.stem,
                "file": test_file.name,
                "passed": False,
                "error": str(e)
            })

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE TEST SUMMARY")
    print("=" * 80)
    print()

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print()

    print("Results:")
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {status} - {r['name']}")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review the manual verification instructions for each test")
    print("2. Check Revenium dashboard for all test events")
    print("3. Document which tests show issues in current state")
    print("4. Use this as baseline to compare after fixes")
    print()
    print("Expected Events in Dashboard:")
    print("  - e2e-test-required-fields (trace: test-required-fields-001)")
    print("  - e2e-test-nested-subscriber (trace: test-nested-subscriber-001)")
    print("  - e2e-test-flat-keys (trace: test-flat-keys-001)")
    print("  - e2e-test-streaming (trace: test-streaming-001)")
    print()
    print("=" * 80)

    # Return 0 if all passed, 1 if any failed
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
