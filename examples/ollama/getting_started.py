import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import ollama
import revenium_middleware.ollama

# Ensure REVENIUM_METERING_API_KEY is set in your .env file
ollama_host = os.getenv("OLLAMA_HOST")
client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()

response = ollama.chat(
    model='qwen2.5:0.5b',
    messages=[
        {
            'role': 'user',
            'content': 'Please verify you are ready to assist me.',
        },
    ],

    # Optional metadata to send to Revenium for
    # advanced reporting. Uncomment lines
    # as needed for your use case.

    # usage_metadata={
        # "trace_id": "conv-28a7e9d4",
        # "task_type": "analyze-spectral-data",
        # "subscriber": {
        #   "id": "1473847563",
        #   "email": "carol@finoptic.com",
        #   "credential": {
        #     "name": "Engineering API Key",
        #     "value": "hak-abc123456"
        #   }
        # },
        # "organizationName": "Finoptic Labs",
        # "subscription_id": "sub_gold_1234567890",
        # "productName": "spectral-analyzer-gold",
        # "agent": "chemistry-agent",
        # "response_quality_score": 0.95,  # Optional: 0.0-1.0 scale quality metric from your evaluation logic
        # "agentic_job_id": "job-abc123",
        # "agentic_job_name": "Process Loan Application",
        # "agentic_job_type": "loan-processing",
        # "agentic_job_version": "1.0.0"
    # },

)
print(response['message']['content'])
