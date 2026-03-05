"""Quick test for Imagen metering."""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import middleware first
import revenium_middleware.google
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Testing Imagen (image generation) with Google AI SDK...")
try:
    response = client.models.generate_images(
        model="imagen-3.0-generate-001",
        prompt="A small red circle on a white background",
        config={"number_of_images": 1},
    )
    print(f"Generated {len(response.generated_images)} image(s)")
    if response.generated_images:
        img = response.generated_images[0]
        img_data = img.image.image_bytes
        print(f"Image size: {len(img_data)} bytes")
    print("PASS - Imagen metering sent successfully")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
