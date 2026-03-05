"""
Example demonstrating prompt capture with Google AI and Vertex AI SDKs.

This example shows how to enable prompt capture to send system prompts,
input messages, and output responses to Revenium for monitoring and analysis.

Prerequisites:
    - Set GOOGLE_API_KEY environment variable for Google AI SDK
    - Set up Google Cloud credentials for Vertex AI SDK
    - Set REVENIUM_METERING_API_KEY environment variable
    - Set REVENIUM_METERING_BASE_URL environment variable (optional)
    - Enable prompt capture: REVENIUM_CAPTURE_PROMPTS=true
"""

import os
import sys

# Enable prompt capture
os.environ['REVENIUM_CAPTURE_PROMPTS'] = 'true'

# Import middleware BEFORE importing Google SDKs (auto-initializes)
import revenium_middleware.google

print("=" * 80)
print("Google AI & Vertex AI Prompt Capture Example")
print("=" * 80)
print()

# Example 1: Google AI SDK (Gemini Developer API)
print("Example 1: Google AI SDK with Prompt Capture")
print("-" * 80)

try:
    from google import genai

    # Configure API key
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not set, skipping Google AI example")
    else:
        client = genai.Client(api_key=api_key)
        model_id = 'gemini-2.0-flash-exp'

        # Create config with system instruction
        config = genai.types.GenerateContentConfig(
            system_instruction="You are a helpful math tutor. Explain concepts clearly and concisely."
        )

        # Non-streaming example
        print("\n📝 Non-streaming request:")
        response = client.models.generate_content(
            model=model_id,
            contents="What is the Pythagorean theorem?",
            config=config
        )
        print(f"Response: {response.text[:100]}...")
        print("✅ Prompt data captured and sent to Revenium")

        # Streaming example
        print("\n📝 Streaming request:")
        print("Response: ", end="", flush=True)
        for chunk in client.models.generate_content_stream(
            model=model_id,
            contents="Explain the quadratic formula in one sentence.",
            config=config
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        print("✅ Streaming prompt data captured and sent to Revenium")

except ImportError:
    print("⚠️  google-genai not installed, skipping Google AI example")
except Exception as e:
    print(f"❌ Error in Google AI example: {e}")

print()

# Example 2: Vertex AI SDK
print("Example 2: Vertex AI SDK with Prompt Capture")
print("-" * 80)

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    # Initialize Vertex AI (requires Google Cloud credentials)
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    
    if not project_id:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set, skipping Vertex AI example")
    else:
        vertexai.init(project=project_id, location=location)
        
        # Create model with system instruction
        model = GenerativeModel(
            'gemini-2.0-flash-exp',
            system_instruction="You are a helpful science tutor. Keep explanations brief."
        )
        
        # Non-streaming example
        print("\n📝 Non-streaming request:")
        response = model.generate_content("What is photosynthesis?")
        print(f"Response: {response.text[:100]}...")
        print("✅ Prompt data captured and sent to Revenium")
        
        # Streaming example
        print("\n📝 Streaming request:")
        print("Response: ", end="", flush=True)
        response_stream = model.generate_content(
            "Explain gravity in one sentence.",
            stream=True
        )
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        print("✅ Streaming prompt data captured and sent to Revenium")
        
except ImportError:
    print("⚠️  vertexai not installed, skipping Vertex AI example")
except Exception as e:
    print(f"❌ Error in Vertex AI example: {e}")

print()
print("=" * 80)
print("Prompt Capture Information")
print("=" * 80)
print("""
The following data is captured and sent to Revenium when REVENIUM_CAPTURE_PROMPTS=true:

1. System Prompt: The system instruction provided to the model
2. Input Messages: The user's prompt/question (contents parameter)
3. Output Response: The model's complete response

Configuration:
- REVENIUM_CAPTURE_PROMPTS: Enable/disable prompt capture (default: false)
- REVENIUM_MAX_PROMPT_LENGTH: Maximum length for each field (default: 10000)

Prompts exceeding the maximum length are truncated with a [TRUNCATED] marker.

Check your Revenium dashboard to see the captured prompts!
""")

