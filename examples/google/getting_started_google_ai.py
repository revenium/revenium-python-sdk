"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Getting Started with Google AI SDK

This is the simplest way to get started with Revenium Google AI middleware.
Just import and start making requests!

Prerequisites:
    export GOOGLE_API_KEY="your-google-api-key"
    export REVENIUM_METERING_API_KEY="your-revenium-api-key"

Installation:
    pip install revenium-python-sdk[google-genai]
"""

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using environment variables

import revenium_middleware.google
from google import genai

# Create client
client = genai.Client()

# Make a simple request
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Please verify you are ready to assist me."
)

# Display response
print(response.text)
print("\n Success! Usage data has been sent to Revenium.")

# Optional: Add metadata for tracking
# Uncomment the example below to include custom metadata:
"""
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is 2+2?",
    usage_metadata={
        # User/subscriber information (nested format recommended)
        "subscriber": {
            "id": "user-12345",
            "email": "user@example.com",
            "credential": {
                "name": "api-key-name",
                "value": "api-key-value"
            }
        },

        # Organization and product tracking (names for lookup/auto-creation)
        "organization_name": "AcmeCorp",
        "product_name": "customer-chatbot",

        # Request identification
        "trace_id": "unique-trace-id",
        "task_type": "question-answering",

        # Agent information
        "agent": "my-assistant",

        # Quality metrics
        "response_quality_score": 0.95,
        
        # Agentic Job tracking
        "agentic_job_id": "job-abc123",
        "agentic_job_name": "Process Loan Application",
        "agentic_job_type": "loan-processing",
        "agentic_job_version": "1.0.0"
    }
)
"""
