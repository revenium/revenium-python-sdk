#!/usr/bin/env python3
"""
 OpenAI Basic Example

Shows how to use Revenium middleware with OpenAI chat completions and embeddings.
Demonstrates seamless metadata integration - all metadata fields are optional!
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.openai.middleware

def openai_basic_example():
    print("OpenAI Basic Usage with Seamless Metadata Integration\n")

    # Verify API keys are set
    openai_key = os.getenv("OPENAI_API_KEY")
    revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

    if not openai_key:
        print("Error: OPENAI_API_KEY not set")
        print("   Add it to your .env file: OPENAI_API_KEY=\"sk-...\"")
        return

    if not revenium_key:
        print("Error: REVENIUM_METERING_API_KEY not set")
        print("   Add it to your .env file: REVENIUM_METERING_API_KEY=\"hak_...\"")
        return

    print(f" OpenAI API Key: {'*' * (len(openai_key) - 4)}{openai_key[-4:]}")
    print(f" Revenium API Key: {'*' * (len(revenium_key) - 4)}{revenium_key[-4:]}")
    print()

    # Create OpenAI client (middleware is automatically active)
    openai_client = OpenAI()

    # Example 1: Basic chat completion (no metadata)
    print("Example 1: Basic chat completion (automatic tracking)")
    
    basic_response = openai_client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": "What is TypeScript in one sentence?"}
        ]
        # No usage_metadata - still automatically tracked!
        # No max_tokens - let response complete naturally
    )

    print(f" Response: {basic_response.choices[0].message.content}")
    print(f" Usage: {basic_response.usage}")
    print("Automatically tracked to Revenium without metadata\n")

    # Example 2: Chat completion with rich metadata (all optional!)
    print("Example 2: Chat completion with rich metadata")
    
    metadata_response = openai_client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": "Explain the benefits of using middleware in 2 sentences."}
        ],
        
        #  All metadata fields are optional - add what you need!
        usage_metadata={
            # User tracking (optional)
            "subscriber": {
                "id": "user-12345",
                "email": "developer@company.com"
            },

            # Business context (optional)
            "organizationName": "AcmeCorp",
            "productName": "customer-chatbot",

            # Task classification (optional)
            "task_type": "explanation-request",
            "trace_id": f"session-{int(time.time() * 1000)}",

            # Custom fields (optional)
            "agent": "openai-python-basic",
            "response_quality_score": 0.95
        }
    )

    print(f" Response: {metadata_response.choices[0].message.content}")
    print(f" Usage: {metadata_response.usage}")
    print("Tracked with rich metadata for analytics\n")

    # Example 3: Basic embeddings (no metadata)
    print("Example 3: Basic embeddings (automatic tracking)")
    
    basic_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input="Revenium middleware automatically tracks OpenAI usage"
        # No usage_metadata - still automatically tracked!
    )

    print(f" Model: {basic_embedding.model}")
    print(f" Usage: {basic_embedding.usage}")
    print(f" Embedding dimensions: {len(basic_embedding.data[0].embedding)}")
    print("Embeddings automatically tracked without metadata\n")

    # Example 4: Embeddings with metadata (all optional!)
    print("Example 4: Embeddings with rich metadata")
    
    metadata_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input="Advanced text embedding with comprehensive tracking metadata",
        
        #  All metadata fields are optional - customize for your use case!
        usage_metadata={
            "subscriber": {
                "id": "embedding-user-789"
            },
            "organizationName": "AcmeCorp",
            "task_type": "document-embedding",
            "productName": "knowledge-base-search",
            "trace_id": f"embed-{int(time.time() * 1000)}"
        }
    )

    print(f" Model: {metadata_embedding.model}")
    print(f" Usage: {metadata_embedding.usage}")
    print(f" Embedding dimensions: {len(metadata_embedding.data[0].embedding)}")
    print("Embeddings tracked with metadata for business analytics\n")

    # Summary
    print("Summary:")
    print("Chat completions work with or without metadata")
    print("Embeddings work with or without metadata")
    print("All metadata fields are optional")
    print("No type casting required - native Python support")
    print("Automatic tracking for all OpenAI API calls")
    print("Rich business analytics when metadata is provided")

if __name__ == "__main__":
    openai_basic_example()
