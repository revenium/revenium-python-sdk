#!/usr/bin/env python3
"""
 OpenAI Streaming Example

Shows how to use Revenium middleware with streaming OpenAI responses and batch embeddings.
Demonstrates seamless metadata integration with streaming - all metadata fields are optional!
"""

import os
import time
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.openai.middleware

def openai_streaming_example():
    print("OpenAI Streaming with Seamless Metadata Integration\n")

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

    # Example 1: Basic streaming (no metadata)
    print("Example 1: Basic streaming chat (automatic tracking)")
    print("Assistant: ", end="", flush=True)
    
    basic_stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Count from 1 to 5 slowly"}
        ],
        stream=True
        # No usage_metadata - still automatically tracked when stream completes!
        # No max_tokens - let response complete naturally
    )

    for chunk in basic_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n Streaming automatically tracked to Revenium without metadata\n")

    # Example 2: Streaming with rich metadata (all optional!)
    print("Example 2: Streaming chat with rich metadata")
    print("Assistant: ", end="", flush=True)
    
    metadata_stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a haiku about middleware"}
        ],
        stream=True,
        
        #  All metadata fields are optional - add what you need for analytics!
        usage_metadata={
            # User tracking (optional)
            "subscriber": {
                "id": "streaming-user-456",
                "email": "poet@company.com"
            },

            # Business context (optional)
            "organizationName": "AcmeCorp",
            "productName": "customer-chatbot",

            # Task classification (optional)
            "task_type": "creative-writing",
            "trace_id": f"stream-{int(time.time() * 1000)}",

            # Custom fields (optional)
            "agent": "openai-python-streaming",
            "response_quality_score": 0.9
        }
    )

    for chunk in metadata_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n Streaming tracked with rich metadata for analytics\n")

    # Example 3: Batch embeddings (no metadata)
    print("Example 3: Batch embeddings (automatic tracking)")
    
    batch_embeddings = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[
            "First document for batch processing",
            "Second document for batch processing",
            "Third document for batch processing"
        ]
        # No usage_metadata - still automatically tracked!
    )

    print(f" Model: {batch_embeddings.model}")
    print(f" Usage: {batch_embeddings.usage}")
    print(f" Embeddings count: {len(batch_embeddings.data)}")
    print("Batch embeddings automatically tracked without metadata\n")

    # Example 4: Embeddings with metadata for batch processing
    print("Example 4: Batch embeddings with metadata")
    
    metadata_batch_embeddings = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[
            "Document 1: Streaming responses provide real-time feedback",
            "Document 2: Metadata enables rich business analytics",
            "Document 3: Batch processing improves efficiency"
        ],
        
        #  All metadata fields are optional - perfect for batch operations!
        usage_metadata={
            "subscriber": {
                "id": "batch-processor-123"
            },
            "organizationName": "AcmeCorp",
            "task_type": "batch-document-embedding",
            "productName": "customer-chatbot",
            "trace_id": f"batch-{int(time.time() * 1000)}",
            "agent": "openai-python-streaming-batch"
        }
    )

    print(f" Model: {metadata_batch_embeddings.model}")
    print(f" Usage: {metadata_batch_embeddings.usage}")
    print(f" Embeddings count: {len(metadata_batch_embeddings.data)}")
    print("Batch embeddings tracked with metadata for business insights\n")

    # Summary
    print("Summary:")
    print("Streaming responses work seamlessly with metadata")
    print("Usage tracked automatically when streams complete")
    print("Batch embeddings supported with optional metadata")
    print("All metadata fields are optional")
    print("No type casting required - native Python support")
    print("Real-time streaming + comprehensive analytics")

if __name__ == "__main__":
    openai_streaming_example()
