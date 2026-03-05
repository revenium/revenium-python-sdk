"""
Getting Started Example

This is the simplest way to get started with Revenium OpenAI middleware.
Just import and start making requests!

Prerequisites:
    Create a .env file in your project directory with:
        OPENAI_API_KEY="your-openai-key"
        REVENIUM_METERING_API_KEY="your-revenium-key"
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

import revenium_middleware.openai.middleware

# Check required environment variables
required_vars = {
    "OPENAI_API_KEY": "sk-...",
    "REVENIUM_METERING_API_KEY": "hak_..."
}

for var_name, example_value in required_vars.items():
    if not os.getenv(var_name):
        print(f"Error: {var_name} not set")
        print(f"   Add it to your .env file: {var_name}=\"{example_value}\"")
        exit(1)

# Create client
client = OpenAI()

# Create metadata (optional - use what you need)
metadata = {
    "organizationName": "AcmeCorp",
    "productName": "customer-chatbot",

    # Additional optional fields (uncomment to use):
    # "subscriber": {
    #     "id": "user-123",
    #     "email": "user@example.com",
    #     "credential": {
    #         "name": "api-key-name",
    #         "value": "api-key-value"
    #     }
    # },
    # "task_type": "chat-completion",
    # "trace_id": "session-abc-123",
    # "agent": "my-agent",
    # "subscription_id": "plan-pro",
    # "response_quality_score": 0.95
}

# Make request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello! Introduce yourself in one sentence."}
    ],
    usage_metadata=metadata
)

# Display response
print(response.choices[0].message.content)
print("\nUsage data sent to Revenium!")
