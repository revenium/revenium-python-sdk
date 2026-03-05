import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import required libraries AFTER loading .env
import requests
import json

# Load config from .env file
proxy_base_url = os.getenv("LITELLM_PROXY_URL")
api_key = os.getenv("LITELLM_API_KEY")

if not proxy_base_url or not api_key:
    print("Error: LITELLM_PROXY_URL or LITELLM_API_KEY not found in .env file.")
    exit()

# Construct the full URL with the specific endpoint needed for this request
proxy_url = f"{proxy_base_url.rstrip('/')}/chat/completions"

print(f"Base URL: {proxy_base_url}")
print(f"Full endpoint URL: {proxy_url}")
print(f"Making request to: {proxy_url}")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",

    # Optional metadata to send to Revenium for
    # advanced reporting (sent as headers)

#    "x-revenium-trace-id": "conv-28a7e9d4",
#    "x-revenium-task-type": "analyze-spectral-data",
#    "x-revenium-organization-name": "Finoptic Labs",
#    "x-revenium-product-name": "spectral-analyzer-gold",
#    "x-revenium-agent": "chemistry-agent",
#    "x-revenium-subscriber-email": "carol@finoptic.com",
#    "x-revenium-subscriber-id": "1473847563",

}

data = {
    "model": "openai/gpt-4o-mini", # Ensure your proxy is configured with this model
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please verify you're ready to assist me."}
    ]
}

try:
    response = requests.post(proxy_url, headers=headers, data=json.dumps(data))
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error calling LiteLLM Proxy: {e}")
except json.JSONDecodeError:
    print("Error decoding JSON response:")
    print(response.text)