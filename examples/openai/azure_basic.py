#!/usr/bin/env python3
"""
 Azure OpenAI Basic Example

Shows how to use Revenium middleware with Azure OpenAI chat completions and embeddings.
Demonstrates seamless metadata integration with Azure - all metadata fields are optional!
"""

import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Import the middleware (this automatically enables the patching)
import revenium_middleware.openai.middleware

def azure_basic_example():
    print("Azure OpenAI Basic Usage with Seamless Metadata Integration\n")

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
    print(f" Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f" API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')}")
    print()

    # Check if we have a chat model configured
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    is_chat_model = deployment and 'embedding' not in deployment.lower()

    if not is_chat_model:
        print("Note: Current Azure deployment appears to be for embeddings.")
        print("To test chat completions, update .env to use a chat model:")
        print("- Comment out the embeddings section")
        print("- Uncomment the chat testing section")
        print("- Set AZURE_OPENAI_DEPLOYMENT=gpt-4o")
        print("\n   Skipping chat examples and testing embeddings instead...\n")
    else:
        # Example 1: Basic Azure chat completion (no metadata)
        print("Example 1: Basic Azure chat completion (automatic tracking)")

        basic_response = azure.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": "What are the benefits of using Azure OpenAI?"}
            ]
            # No usage_metadata - still automatically tracked with Azure provider info!
            # No max_tokens - let response complete naturally
        )

        print(f" Response: {basic_response.choices[0].message.content}")
        print(f" Usage: {basic_response.usage}")
        print("Automatically tracked to Revenium with Azure provider metadata\n")

    if is_chat_model:
        # Example 2: Azure chat completion with rich metadata (all optional!)
        print("Example 2: Azure chat completion with rich metadata")

        metadata_response = azure.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": "Explain how Azure OpenAI differs from standard OpenAI in 3 points."}
            ],

            #  All metadata fields are optional - add what you need for Azure analytics!
            usage_metadata={
                # User tracking (optional)
                "subscriber": {
                    "id": "azure-user-789",
                    "email": "azure-dev@company.com"
                },

                # Business context (optional)
                "organizationName": "AcmeCorp",
                "productName": "customer-chatbot",

                # Task classification (optional)
                "task_type": "azure-comparison",
                "trace_id": f"azure-{int(time.time() * 1000)}",

                # Azure-specific fields (optional)
                "agent": "azure-openai-python-basic",
                "response_quality_score": 0.92
            }
        )

        print(f" Response: {metadata_response.choices[0].message.content}")
        print(f" Usage: {metadata_response.usage}")
        print("Tracked with Azure provider + rich metadata for enterprise analytics\n")

    # Example 3: Azure embeddings (requires embeddings model)
    print("Example 3: Azure embeddings")

    if is_chat_model:
        print("Note: Current deployment is a chat model (gpt-4o).")
        print("Embeddings require an embedding model like text-embedding-3-large.")
        print("To test embeddings, switch .env to embeddings configuration.")
        print("Skipping embeddings examples.\n")
    else:
        print("Example 3a: Basic Azure embeddings (automatic tracking)")

        basic_embedding = azure.embeddings.create(
            model=deployment or "text-embedding-3-large",
            input="Azure OpenAI provides enterprise-grade AI capabilities with enhanced security and compliance"
            # No usage_metadata - still automatically tracked with Azure provider info!
        )

        print(f" Model: {basic_embedding.model}")
        print(f" Usage: {basic_embedding.usage}")
        print(f" Embedding dimensions: {len(basic_embedding.data[0].embedding)}")
        print("Azure embeddings automatically tracked without metadata\n")

        # Example 4: Azure embeddings with metadata (all optional!)
        print("Example 3b: Azure embeddings with rich metadata")

        metadata_embedding = azure.embeddings.create(
            model=deployment or "text-embedding-3-large",
            input="Enterprise document processing with Azure OpenAI embeddings and comprehensive tracking",

            #  All metadata fields are optional - perfect for Azure enterprise use cases!
            usage_metadata={
                "subscriber": {
                    "id": "azure-embed-user-456"
                },
                "organizationName": "AcmeCorp",
                "task_type": "enterprise-document-embedding",
                "productName": "customer-chatbot",
                "trace_id": f"azure-embed-{int(time.time() * 1000)}",
                "agent": "azure-document-processor"
            }
        )

        print(f" Model: {metadata_embedding.model}")
        print(f" Usage: {metadata_embedding.usage}")
        print(f" Embedding dimensions: {len(metadata_embedding.data[0].embedding)}")
        print("Azure embeddings tracked with metadata for enterprise analytics\n")

    # Summary
    print("Azure OpenAI Summary:")
    print("Azure OpenAI automatically detected and tracked")
    print("Model name resolution for accurate pricing")
    print("Provider metadata includes 'Azure' for analytics")
    if is_chat_model:
        print("Chat completions work with or without metadata")
    else:
        print("Embeddings work with or without metadata")
    print("All metadata fields are optional")
    print("No type casting required - native Python support")
    print("Enterprise-grade tracking with Azure compliance")

    if not is_chat_model:
        print("\n To test Azure chat completions:")
        print("1. Edit .env file")
        print("2. Comment out embeddings section")
        print("3. Uncomment chat section")
        print("4. Run this example again")
    else:
        print("\n To test Azure embeddings:")
        print("1. Edit .env file")
        print("2. Comment out chat section")
        print("3. Uncomment embeddings section")
        print("4. Run this example again")

if __name__ == "__main__":
    azure_basic_example()
