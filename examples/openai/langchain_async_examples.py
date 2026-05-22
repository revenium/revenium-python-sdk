"""
LangChain Async Examples for Revenium Middleware

This file demonstrates various async usage patterns with LangChain and Revenium middleware.
All examples include automatic usage tracking and error handling.

Prerequisites:
    pip install revenium-python-sdk[openai,langchain]

    Create a .env file in your project directory with:
        OPENAI_API_KEY="your-openai-key"
        REVENIUM_METERING_API_KEY="your-revenium-key"
"""

import asyncio
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    # LangChain 1.0+ uses langchain_core
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    # LangChain 0.x uses langchain.schema
    from langchain.schema import HumanMessage, SystemMessage

# Revenium middleware imports
from revenium_middleware.openai.langchain import wrap, attach_to


async def basic_async_chat_example():
    """Basic async chat completion with automatic usage tracking."""
    print("Basic Async Chat Example")
    
    # Wrap LangChain LLM with Revenium tracking
    llm = wrap(
        ChatOpenAI(model="gpt-4o-mini"),
        usage_metadata={
            "trace_id": "async-example-001",
            "task_type": "basic-chat",
            "agent": "langchain-python-openai-basic"
        }
    )
    
    # Use async invoke - usage is automatically tracked
    response = await llm.ainvoke("What are the benefits of async programming?")
    print(f"Response: {response.content[:100]}...")
    print("Basic async chat completed\n")


async def async_streaming_example():
    """Async streaming with real-time token processing."""
    print("Async Streaming Example")
    
    # Wrap streaming LLM
    llm = wrap(
        ChatOpenAI(model="gpt-4o-mini", streaming=True),
        usage_metadata={
            "trace_id": "async-stream-001",
            "task_type": "streaming-chat"
        }
    )
    
    print("Streaming response: ", end="")
    
    # Stream tokens asynchronously
    async for chunk in llm.astream("Tell me a short story about async programming"):
        print(chunk.content, end="", flush=True)
    
    print("\n Async streaming completed\n")


async def async_embeddings_example():
    """Async embeddings generation with batch processing."""
    print("Async Embeddings Example")
    
    # Wrap embeddings model
    embeddings = wrap(
        OpenAIEmbeddings(model="text-embedding-3-small"),
        usage_metadata={
            "trace_id": "async-embed-001",
            "task_type": "document-embedding"
        }
    )
    
    # Generate embeddings asynchronously
    texts = [
        "Async programming enables concurrent execution",
        "LangChain provides async support for LLM operations",
        "Revenium tracks usage across sync and async calls"
    ]
    
    vectors = await embeddings.aembed_documents(texts)
    print(f"Generated {len(vectors)} embeddings with {len(vectors[0])} dimensions each")
    print("Async embeddings completed\n")


async def concurrent_operations_example():
    """Multiple concurrent async operations with usage tracking."""
    print("Concurrent Operations Example")
    
    # Create multiple LLMs for concurrent operations
    chat_llm = wrap(
        ChatOpenAI(model="gpt-4o-mini"),
        usage_metadata={"task_type": "concurrent-chat"}
    )
    
    embedding_model = wrap(
        OpenAIEmbeddings(model="text-embedding-3-small"),
        usage_metadata={"task_type": "concurrent-embedding"}
    )
    
    # Define concurrent tasks
    async def chat_task():
        return await chat_llm.ainvoke("What is concurrency?")
    
    async def embedding_task():
        return await embedding_model.aembed_query("Concurrency in programming")
    
    async def streaming_task():
        stream_llm = wrap(ChatOpenAI(model="gpt-4o-mini", streaming=True))
        chunks = []
        async for chunk in stream_llm.astream("Explain async/await"):
            chunks.append(chunk.content)
        return "".join(chunks)
    
    # Run tasks concurrently
    chat_result, embedding_result, stream_result = await asyncio.gather(
        chat_task(),
        embedding_task(),
        streaming_task()
    )
    
    print(f"Chat result: {chat_result.content[:50]}...")
    print(f"Embedding dimensions: {len(embedding_result)}")
    print(f"Stream result: {stream_result[:50]}...")
    print("Concurrent operations completed\n")


async def error_handling_example():
    """Async error handling with graceful degradation."""
    print("Error Handling Example")
    
    # Create LLM with error-prone configuration
    llm = wrap(
        ChatOpenAI(model="gpt-4o-mini", max_tokens=1),  # Very low token limit
        usage_metadata={"task_type": "error-handling-test"}
    )
    
    try:
        # This might hit token limits
        response = await llm.ainvoke("Write a very long essay about async programming")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Handled error: {type(e).__name__}: {str(e)[:100]}...")
    
    print("Error handling completed\n")


async def streaming_with_metadata_example():
    """Streaming with comprehensive metadata tracking."""
    print("Streaming with Metadata Example")

    # Create streaming LLM with metadata
    llm = wrap(
        ChatOpenAI(model="gpt-4o-mini", streaming=True),
        usage_metadata={
            "trace_id": "stream-metadata-001",
            "task_type": "streaming-count",
            "subscriber": {
                "id": "stream-user-456"
            },
            "agent": "streaming-demo"
        }
    )

    # Start streaming operation
    print("Streaming response: ", end="")
    async for chunk in llm.astream("Count from 1 to 10 slowly"):
        print(chunk.content, end="", flush=True)

    print("\n Streaming with metadata completed\n")


async def advanced_configuration_example():
    """Advanced configuration with custom metadata and settings."""
    print("Advanced Configuration Example")
    
    # Method 1: Using wrap() with comprehensive metadata
    llm1 = wrap(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=150
        ),
        usage_metadata={
            "trace_id": "advanced-config-001",
            "task_type": "creative-writing",
            "subscriber": {
                "id": "user-12345"
            },
            "organizationName": "AcmeCorp",
            "productName": "content-generator",
            "agent": "langchain-python-openai-advanced"
        },
        enable_debug_logging=True
    )
    
    # Method 2: Using attach_to() for in-place modification
    llm2 = ChatOpenAI(model="gpt-4o-mini")
    attach_to(
        llm2,
        usage_metadata={
            "trace_id": "advanced-config-002",
            "task_type": "analysis"
        }
    )
    
    # Use both LLMs
    response1 = await llm1.ainvoke("Write a creative short story opening")
    response2 = await llm2.ainvoke("Analyze the benefits of async programming")
    
    print(f"Creative response: {response1.content[:100]}...")
    print(f"Analysis response: {response2.content[:100]}...")
    print("Advanced configuration completed\n")


async def main():
    """Run all async examples."""
    print("LangChain Async Examples with Revenium Middleware\n")

    # Check environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not set")
        print("   Add it to your .env file: OPENAI_API_KEY=\"sk-...\"")
        return

    if not os.getenv('REVENIUM_METERING_API_KEY'):
        print("Error: REVENIUM_METERING_API_KEY not set")
        print("   Add it to your .env file: REVENIUM_METERING_API_KEY=\"hak_...\"")
        return
    
    # Run examples
    examples = [
        basic_async_chat_example,
        async_streaming_example,
        async_embeddings_example,
        concurrent_operations_example,
        error_handling_example,
        streaming_with_metadata_example,
        advanced_configuration_example
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f" Error in {example.__name__}: {e}\n")
    
    print("All async examples completed!")


if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main())
