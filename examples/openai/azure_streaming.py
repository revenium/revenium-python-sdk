#!/usr/bin/env python3
"""
 Azure OpenAI Streaming Example

Shows how to use Revenium middleware with Azure OpenAI streaming responses.
Demonstrates seamless metadata integration with Azure streaming - all metadata fields are optional!
"""

import os
import time
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.openai.middleware 

def azure_streaming_example():
    print("Azure OpenAI Streaming with Seamless Metadata Integration\n")

    # Verify API keys are set
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    revenium_key = os.getenv("REVENIUM_METERING_API_KEY")

    if not azure_endpoint:
        print("Error: AZURE_OPENAI_ENDPOINT not set")
        print("   Add it to your .env file: AZURE_OPENAI_ENDPOINT=\"https://your-resource.openai.azure.com/\"")
        return

    if not azure_key:
        print("Error: AZURE_OPENAI_API_KEY not set")
        print("   Add it to your .env file: AZURE_OPENAI_API_KEY=\"your_azure_key_here\"")
        return

    if not revenium_key:
        print("Error: REVENIUM_METERING_API_KEY not set")
        print("   Add it to your .env file: REVENIUM_METERING_API_KEY=\"hak_...\"")
        return

    # Create Azure OpenAI client (middleware is automatically active)
    azure = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    )

    print("Azure OpenAI client configured and patched")
    print(f" Endpoint: {azure_endpoint}")
    print(f" API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')}")
    print()

    # Check if we have a chat model configured
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    is_chat_model = deployment and 'embedding' not in deployment.lower()

    if not is_chat_model:
        print("Note: Current Azure deployment appears to be for embeddings.")
        print("To test streaming chat, update .env to use a chat model:")
        print("- Comment out the embeddings section")
        print("- Uncomment the chat testing section")
        print("- Set AZURE_OPENAI_DEPLOYMENT=gpt-4o")
        print("\n   Testing embeddings instead (no streaming for embeddings)...\n")
    else:
        # Example 1: Basic Azure streaming (no metadata)
        print("Example 1: Basic Azure streaming chat (automatic tracking)")
        print("Assistant: ", end="", flush=True)

        basic_stream = azure.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": "List 3 advantages of Azure OpenAI over standard OpenAI"}
            ],
            stream=True
            # No usage_metadata - still automatically tracked with Azure provider info when stream completes!
            # No max_tokens - let response complete naturally
        )

        for chunk in basic_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n Azure streaming automatically tracked to Revenium without metadata\n")

        # Example 2: Azure streaming with rich metadata (all optional!)
        print("Example 2: Azure streaming chat with rich metadata")
        print("Assistant: ", end="", flush=True)

        metadata_stream = azure.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": "Write a professional summary about Azure OpenAI benefits for enterprises"}
            ],
            stream=True,

            #  All metadata fields are optional - add what you need for Azure enterprise analytics!
            usage_metadata={
                # User tracking (optional)
                "subscriber": {
                    "id": "azure-stream-user-789",
                    "email": "enterprise@company.com"
                },

                # Business context (optional)
                "organizationName": "AcmeCorp",
                "productName": "customer-chatbot",

                # Task classification (optional)
                "task_type": "enterprise-consultation",
                "trace_id": f"azure-stream-{int(time.time() * 1000)}",

                # Azure-specific fields (optional)
                "agent": "azure-openai-python-streaming",
                "response_quality_score": 0.95
            }
        )

        for chunk in metadata_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n Azure streaming tracked with rich metadata for enterprise analytics\n")

    # Example 3: Azure batch embeddings (no metadata)
    print("Example 3: Azure batch embeddings (automatic tracking)")
    
    batch_embeddings = azure.embeddings.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT") or "text-embedding-3-large",
        input=[
            "Azure OpenAI provides enterprise security and compliance",
            "Private network access ensures data protection",
            "Managed identity integration simplifies authentication"
        ]
        # No usage_metadata - still automatically tracked with Azure provider info!
    )

    print(f" Model: {batch_embeddings.model}")
    print(f" Usage: {batch_embeddings.usage}")
    print(f" Embeddings count: {len(batch_embeddings.data)}")
    print("Azure batch embeddings automatically tracked without metadata\n")

    # Example 4: Azure embeddings with enterprise metadata
    print("Example 4: Azure batch embeddings with enterprise metadata")
    
    enterprise_embeddings = azure.embeddings.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT") or "text-embedding-3-large",
        input=[
            "Enterprise document: Azure OpenAI compliance framework",
            "Enterprise document: Data residency and sovereignty requirements",
            "Enterprise document: Integration with Azure Active Directory"
        ],
        
        #  All metadata fields are optional - perfect for Azure enterprise document processing!
        usage_metadata={
            "subscriber": {
                "id": "azure-enterprise-processor"
            },
            "organizationName": "AcmeCorp",
            "task_type": "enterprise-document-processing",
            "productName": "customer-chatbot",
            "trace_id": f"azure-enterprise-{int(time.time() * 1000)}",
            "agent": "azure-document-analyzer"
        }
    )

    print(f" Model: {enterprise_embeddings.model}")
    print(f" Usage: {enterprise_embeddings.usage}")
    print(f" Embeddings count: {len(enterprise_embeddings.data)}")
    print("Azure enterprise embeddings tracked with comprehensive metadata\n")

    # Summary
    print("Azure OpenAI Streaming Summary:")
    print("Azure streaming responses work seamlessly with metadata")
    print("Usage tracked automatically when Azure streams complete")
    print("Azure batch embeddings supported with optional metadata")
    print("Enterprise-grade tracking with Azure provider metadata")
    print("Model name resolution for accurate Azure pricing")
    print("All metadata fields are optional")
    print("No type casting required - native Python support")
    print("Real-time Azure streaming + comprehensive enterprise analytics")

if __name__ == "__main__":
    azure_streaming_example()
