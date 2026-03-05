"""
Embeddings Example

This example demonstrates embeddings usage with Ollama middleware.
The middleware automatically tracks your Ollama embeddings API usage in Revenium.

Run this example:
    python embeddings_example.py

Expected output:
    - Embeddings for single and batch inputs
    - Automatic usage tracking in Revenium
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import ollama
import revenium_middleware.ollama

# Ensure REVENIUM_METERING_API_KEY is set in your .env file
if not os.getenv("REVENIUM_METERING_API_KEY"):
    raise ValueError(
        "REVENIUM_METERING_API_KEY environment variable is not set. "
        "Please set it in your .env file."
    )

print("=" * 80)
print("EMBEDDINGS EXAMPLE")
print("=" * 80)
print()

# Example 1: Single text embedding
print("Example 1: Single Text Embedding")
print("-" * 80)
text = "The quick brown fox jumps over the lazy dog"
print(f"Input text: {text}")

response = ollama.embed(
    model='nomic-embed-text',
    input=text,
)

# Extract and display the embedding
embedding = response['embeddings'][0]
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
print(f"Transaction ID: {response._revenium_transaction_id}")
print()

# Example 2: Batch embeddings
print("Example 2: Batch Embeddings")
print("-" * 80)
texts = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing enables computers to understand text"
]
print(f"Number of texts: {len(texts)}")
for i, text in enumerate(texts, 1):
    print(f"  {i}. {text}")

response = ollama.embed(
    model='nomic-embed-text',
    input=texts,
)

# Extract and display the embeddings
embeddings = response['embeddings']
print(f"\nGenerated {len(embeddings)} embeddings")
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"Transaction ID: {response._revenium_transaction_id}")
print()

# Example 3: Embeddings with usage metadata
print("Example 3: Embeddings with Usage Metadata")
print("-" * 80)
text = "Embeddings convert text into numerical vectors"
print(f"Input text: {text}")

response = ollama.embed(
    model='nomic-embed-text',
    input=text,
    usage_metadata={
        "organizationName": "AcmeCorp",
        "subscription_id": "sub-456",
        "productName": "semantic-search-engine",
        "subscriber": {
            "id": "user-001",
            "email": "user@example.com"
        },
        "trace_id": "trace-embeddings-001",
        "task_type": "text-embedding"
    }
)

embedding = response['embeddings'][0]
print(f"Embedding dimensions: {len(embedding)}")
print(f"Transaction ID: {response._revenium_transaction_id}")
print()

print("All embeddings usage is automatically tracked in Revenium dashboard.")
print()
print("=" * 80)
