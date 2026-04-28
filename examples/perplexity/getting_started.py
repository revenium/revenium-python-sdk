"""
Getting Started Example

This example demonstrates the simplest way to use the Revenium Perplexity middleware.
Just import the middleware and use the OpenAI client with Perplexity's base URL.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import Revenium middleware (this automatically patches OpenAI)
import revenium_middleware.perplexity  # noqa: F401

def main():
    """Run a simple Perplexity chat completion with automatic metering."""
    
    # Create OpenAI client with Perplexity base URL
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    print("Sending request to Perplexity...")
    
    # Make a chat completion request
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one sentence."
            }
        ],

        # Optional metadata for advanced reporting, lineage tracking, and cost allocation
        # usage_metadata={
        #     "subscriber": {
        #         "id": "user-123",
        #         "email": "user@example.com",
        #         "credential": {
        #             "name": "api-key-prod",
        #             "value": "key-abc-123"
        #         }
        #     },
        #     "organization_name": "AcmeCorp",
        #     "subscription_id": "plan-enterprise-2024",
        #     "product_name": "customer-chatbot",
        #     "task_type": "question-answering",
        #     "agent": "research-agent",
        #     "trace_id": "session-abc123",
        #     "response_quality_score": 0.95,
        #     "agentic_job_id": "job-abc123",
        #     "agentic_job_name": "Process Loan Application",
        #     "agentic_job_type": "loan-processing",
        #     "agentic_job_version": "1.0.0"
        # }
    )

    # Display the response
    print(f"\nAssistant: {response.choices[0].message.content}")
    print(f"\nTokens used: {response.usage.total_tokens}")
    print("\nUsage data automatically sent to Revenium!")


if __name__ == "__main__":
    main()

