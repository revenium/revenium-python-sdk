"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Getting Started with Vertex AI SDK

This is the simplest way to get started with Revenium Vertex AI middleware.
Just import and start making requests!

Prerequisites:
    export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    export REVENIUM_METERING_API_KEY="your-revenium-api-key"

    # Authentication (one of):
    # - gcloud auth application-default login
    # - export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

Installation:
    pip install revenium-python-sdk[google-vertex]
"""

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using environment variables

import revenium_middleware.google
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
vertexai.init()

# Create model
model = GenerativeModel("gemini-2.0-flash-001")

# Make a simple request
response = model.generate_content("Please verify you are ready to assist me.")

# Display response
print(response.text)
print("\n Success! Usage data has been sent to Revenium.")

# Optional: Add metadata for tracking
# Uncomment the example below to include custom metadata:
"""
# Set metadata on model instance
model._revenium_usage_metadata = {
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
    "organizationName": "AcmeCorp",
    "productName": "customer-chatbot",

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

response = model.generate_content("What is 2+2?")
"""
