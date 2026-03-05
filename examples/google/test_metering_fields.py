"""
Direct Revenium metering API calls to verify new fields:
- has_vision_content on create_completion
- aspect_ratio on create_image
- aspect_ratio on create_video
"""
import os
import uuid
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from revenium_metering import ReveniumMetering

client = ReveniumMetering(
    api_key=os.getenv("REVENIUM_METERING_API_KEY"),
    base_url=os.getenv("REVENIUM_METERING_BASE_URL"),
)

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
results = {}

# --- Test 1: create_completion with has_vision_content ---
print("=" * 60)
print("TEST 1: create_completion with has_vision_content=True")
print("=" * 60)
try:
    result = client.ai.create_completion(
        model="gemini-2.0-flash-001",
        provider="Google",
        input_token_count=269,
        output_token_count=1,
        total_token_count=270,
        cost_type="AI",
        request_duration=2000,
        request_time=now,
        response_time=now,
        completion_start_time=now,
        stop_reason="END",
        transaction_id=str(uuid.uuid4()),
        is_streamed=False,
        has_vision_content=True,
        middleware_source="test-script",
    )
    print(f"  PASS - id: {result.id}")
    results["completion_with_vision"] = True
except Exception as e:
    print(f"  FAIL - {e}")
    results["completion_with_vision"] = False

# --- Test 2: create_completion WITHOUT has_vision_content (baseline) ---
print()
print("=" * 60)
print("TEST 2: create_completion without has_vision_content (baseline)")
print("=" * 60)
try:
    result = client.ai.create_completion(
        model="gemini-2.0-flash-001",
        provider="Google",
        input_token_count=100,
        output_token_count=50,
        total_token_count=150,
        cost_type="AI",
        request_duration=1000,
        request_time=now,
        response_time=now,
        completion_start_time=now,
        stop_reason="END",
        transaction_id=str(uuid.uuid4()),
        is_streamed=False,
        middleware_source="test-script",
    )
    print(f"  PASS - id: {result.id}")
    results["completion_baseline"] = True
except Exception as e:
    print(f"  FAIL - {e}")
    results["completion_baseline"] = False

# --- Test 3: create_image with aspect_ratio ---
print()
print("=" * 60)
print("TEST 3: create_image with aspect_ratio='16:9'")
print("=" * 60)
try:
    result = client.ai.create_image(
        model="imagen-3.0-generate-001",
        provider="Google",
        request_duration=5000,
        request_time=now,
        response_time=now,
        transaction_id=str(uuid.uuid4()),
        requested_image_count=1,
        actual_image_count=1,
        resolution="1024x1024",
        aspect_ratio="16:9",
        middleware_source="test-script",
    )
    print(f"  PASS - id: {result.id}")
    results["image_with_aspect_ratio"] = True
except Exception as e:
    print(f"  FAIL - {e}")
    results["image_with_aspect_ratio"] = False

# --- Test 4: create_image WITHOUT aspect_ratio (baseline) ---
print()
print("=" * 60)
print("TEST 4: create_image without aspect_ratio (baseline)")
print("=" * 60)
try:
    result = client.ai.create_image(
        model="imagen-3.0-generate-001",
        provider="Google",
        request_duration=5000,
        request_time=now,
        response_time=now,
        transaction_id=str(uuid.uuid4()),
        requested_image_count=1,
        actual_image_count=1,
        resolution="1024x1024",
        middleware_source="test-script",
    )
    print(f"  PASS - id: {result.id}")
    results["image_baseline"] = True
except Exception as e:
    print(f"  FAIL - {e}")
    results["image_baseline"] = False

# --- Test 5: create_video with aspect_ratio ---
print()
print("=" * 60)
print("TEST 5: create_video with aspect_ratio='9:16'")
print("=" * 60)
try:
    result = client.ai.create_video(
        model="veo-2.0-generate-001",
        provider="Google",
        request_duration=30000,
        request_time=now,
        response_time=now,
        transaction_id=str(uuid.uuid4()),
        duration_seconds=5.0,
        resolution="1080p",
        aspect_ratio="9:16",
        async_operation=True,
        middleware_source="test-script",
    )
    print(f"  PASS - id: {result.id}")
    results["video_with_aspect_ratio"] = True
except Exception as e:
    print(f"  FAIL - {e}")
    results["video_with_aspect_ratio"] = False

# --- Test 6: create_video WITHOUT aspect_ratio (baseline) ---
print()
print("=" * 60)
print("TEST 6: create_video without aspect_ratio (baseline)")
print("=" * 60)
try:
    result = client.ai.create_video(
        model="veo-2.0-generate-001",
        provider="Google",
        request_duration=30000,
        request_time=now,
        response_time=now,
        transaction_id=str(uuid.uuid4()),
        duration_seconds=5.0,
        resolution="1080p",
        middleware_source="test-script",
    )
    print(f"  PASS - id: {result.id}")
    results["video_baseline"] = True
except Exception as e:
    print(f"  FAIL - {e}")
    results["video_baseline"] = False

# --- Summary ---
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for v in results.values() if v)
total = len(results)
for name, ok in results.items():
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: {name}")
print(f"\n{passed}/{total} passed")
