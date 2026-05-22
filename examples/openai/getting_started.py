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
    "organization_name": "AcmeCorp",
    "product_name": "customer-chatbot",

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
    # "response_quality_score": 0.95,
    # "agentic_job_id": "job-abc123",
    # "agentic_job_name": "Process Loan Application",
    # "agentic_job_type": "loan-processing",
    # "agentic_job_version": "1.0.0"
}

# Make request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful technical writing assistant. When the user "
                "gives you a paragraph of text, produce (a) a concise 2-3 "
                "sentence summary written in plain English, and (b) a bullet "
                "list of the 3 most important points. Keep the tone neutral "
                "and professional. Do not invent facts that are not in the "
                "source paragraph. If the paragraph is ambiguous, prefer the "
                "most literal interpretation and note the ambiguity briefly "
                "at the end of your summary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the following paragraph.\n\n"
                "Modern distributed systems rely heavily on observability to "
                "remain operable at scale. Observability is typically broken "
                "down into three pillars: metrics, logs, and traces. Metrics "
                "are numeric measurements aggregated over time and are well "
                "suited to dashboards and alerting because they are cheap to "
                "store and query. Logs are discrete event records that "
                "capture rich context but are expensive to retain at high "
                "cardinality. Distributed traces stitch together the path of "
                "a single request as it crosses service boundaries, making "
                "them invaluable for diagnosing latency and correctness "
                "issues that no single service can explain on its own. "
                "Mature platforms combine all three so that operators can "
                "move fluidly between a high-level alert, the underlying "
                "events that caused it, and the precise request path that "
                "produced the failure."
            ),
        },
    ],
    max_tokens=400,
    usage_metadata=metadata,
)

# Display response
print(response.choices[0].message.content)
print("\nUsage data sent to Revenium!")
