"""Quick test for Vision metering with has_vision_content flag."""
import base64
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
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a tiny 1x1 red PNG image inline (valid PNG)
import struct, zlib
def make_tiny_png():
    """Create a minimal 1x1 red pixel PNG."""
    sig = b'\x89PNG\r\n\x1a\n'
    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
    ihdr = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + ihdr_crc
    # IDAT chunk - raw pixel data: filter byte (0) + R G B
    raw = zlib.compress(b'\x00\xff\x00\x00')  # filter=0, red pixel
    idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + raw) & 0xffffffff)
    idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + idat_crc
    # IEND chunk
    iend_crc = struct.pack('>I', zlib.crc32(b'IEND') & 0xffffffff)
    iend = struct.pack('>I', 0) + b'IEND' + iend_crc
    return sig + ihdr + idat + iend

png_bytes = make_tiny_png()
b64_image = base64.b64encode(png_bytes).decode()

print("Testing Vision (inline image) with Gemini...")
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[
            types.Content(
                parts=[
                    types.Part(text="What color is this image? Answer in one word."),
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                ]
            )
        ],
    )
    print(f"Response: {response.text[:200]}")
    if response.usage_metadata:
        u = response.usage_metadata
        print(f"Tokens: {u.prompt_token_count} in + {u.candidates_token_count} out = {u.total_token_count} total")
    print("PASS - Vision metering sent successfully (has_vision_content=True)")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
