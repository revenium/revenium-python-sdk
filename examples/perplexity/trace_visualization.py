"""
Trace Visualization Example

This example demonstrates how to use trace visualization fields
for distributed tracing and analytics. These fields are automatically
captured from environment variables.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import Revenium middleware
import revenium_middleware.perplexity  # noqa: F401

def main():
    """
    Run a Perplexity chat completion with trace visualization fields.
    
    The middleware automatically captures these fields from environment variables:
    - REVENIUM_ENVIRONMENT (or NODE_ENV/PYTHON_ENV)
    - REVENIUM_REGION (or AWS_REGION)
    - REVENIUM_CREDENTIAL_ALIAS
    - REVENIUM_TRACE_TYPE
    - REVENIUM_TRACE_NAME
    - REVENIUM_PARENT_TRANSACTION_ID
    - REVENIUM_TRANSACTION_NAME
    - REVENIUM_RETRY_NUMBER
    """
    
    # Set trace visualization fields (normally set in .env or deployment config)
    os.environ["REVENIUM_ENVIRONMENT"] = "production"
    os.environ["REVENIUM_REGION"] = "us-east-1"
    os.environ["REVENIUM_CREDENTIAL_ALIAS"] = "Perplexity Production Key"
    os.environ["REVENIUM_TRACE_TYPE"] = "customer_support"
    os.environ["REVENIUM_TRACE_NAME"] = "Support Ticket #12345"
    os.environ["REVENIUM_TRANSACTION_NAME"] = "Answer Customer Question"
    os.environ["REVENIUM_RETRY_NUMBER"] = "0"
    
    # Create OpenAI client with Perplexity base URL
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    print("Sending request with trace visualization fields...")
    
    # Make a chat completion request
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {
                "role": "user",
                "content": "How do I reset my password?"
            }
        ],
        usage_metadata={
            "organization_id": "org-support",
            "product_id": "prod-helpdesk",
            "subscriber": {
                "id": "customer-789",
                "email": "customer@example.com"
            }
        }
    )
    
    # Display the response
    print(f"\nAssistant: {response.choices[0].message.content}")
    print(f"\nTokens used: {response.usage.total_tokens}")
    print("\n✅ Usage data with trace fields sent to Revenium!")
    print("\nTrace fields included:")
    print(f"  - Environment: {os.getenv('REVENIUM_ENVIRONMENT')}")
    print(f"  - Region: {os.getenv('REVENIUM_REGION')}")
    print(f"  - Trace Type: {os.getenv('REVENIUM_TRACE_TYPE')}")
    print(f"  - Trace Name: {os.getenv('REVENIUM_TRACE_NAME')}")


if __name__ == "__main__":
    main()

