#!/usr/bin/env python3
"""
Getting Started with Revenium Middleware for Anthropic

The simplest example to get you started with Revenium tracking.
Shows all optional metadata fields documented for reference.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def main():
    """Simple example showing automatic usage tracking."""

    import anthropic
    import revenium_middleware.anthropic

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Please verify you are ready to assist me."
            }
        ]

        # Optional metadata for advanced reporting, lineage tracking, and cost allocation
        # usage_metadata={
        #     # User identification
        #     "subscriber": {
        #         "id": "user-123",
        #         "email": "user@example.com",
        #         "credential": {
        #             "name": "api-key-prod",
        #             "value": "key-abc-123"
        #         }
        #     },
        #
        #     # Organization & billing
        #     "organization_name": "AcmeCorp",
        #     "subscription_id": "plan-enterprise-2024",
        #
        #     # Product & task tracking
        #     "product_name": "customer-chatbot",
        #     "task_type": "doc-summary",
        #     "agent": "customer-support",
        #
        #     # Session tracking
        #     "trace_id": "session-abc123",
        #
        #     # Quality metrics
        #     "response_quality_score": 0.95,  # 0.0-1.0 scale
        #
        #     # Agentic Job tracking
        #     "agentic_job_id": "job-abc123",
        #     "agentic_job_name": "Process Loan Application",
        #     "agentic_job_type": "loan-processing",
        #     "agentic_job_version": "1.0.0"
        #
        # }
    )

    print(f"Response: {message.content[0].text}")


if __name__ == "__main__":
    main()
