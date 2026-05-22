"""
Example demonstrating tool metering with the OpenAI middleware.

This example shows how to use the @meter_tool decorator to meter arbitrary
tool/function calls (web scrapers, image generators, database lookups, etc.)
alongside your automatic LLM API metering.

Run this example:
    python examples/example_meter_tool.py

Prerequisites:
    export OPENAI_API_KEY="your-openai-api-key"
    export REVENIUM_METERING_API_KEY="your-revenium-api-key"
    export REVENIUM_METERING_BASE_URL="https://api.revenium.ai"
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from openai import OpenAI

# Import tool metering utilities
from revenium_middleware import meter_tool, configure

# Configure the metering client BEFORE importing middleware
configure(
    metering_url=os.getenv("REVENIUM_METERING_BASE_URL", "https://api.revenium.ai"),
    api_key=os.environ["REVENIUM_METERING_API_KEY"],  # Fail fast if not set
)

# NOW import middleware to enable automatic LLM metering
import revenium_middleware.openai.middleware

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # Fail fast if not set


@meter_tool("web-scraper", operation="scrape", agent="research-assistant")
def scrape_website(url: str) -> dict:
    """
    Simulated web scraper tool.

    The @meter_tool decorator automatically captures:
    - Execution time (duration_ms)
    - Success/failure status
    - Attribution metadata (agent, organization, etc.)
    """
    print(f"  Scraping: {url}")
    # Simulate scraping work
    time.sleep(0.5)
    return {
        "url": url,
        "title": "Example Page - AI News",
        "content": (
            "Artificial intelligence continues to transform industries. "
            "Recent advances in large language models have enabled new applications "
            "in healthcare, finance, and education. Companies are investing heavily "
            "in AI infrastructure to stay competitive."
        ),
        "word_count": 42,
    }


def summarize_with_gpt(text: str) -> str:
    """
    Use GPT to summarize scraped content.

    This call is automatically metered by the OpenAI middleware.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise summarizer. Summarize in one sentence.",
            },
            {"role": "user", "content": f"Summarize this:\n\n{text}"},
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content


def main():
    """Run the tool metering example."""
    print("=" * 60)
    print("OpenAI Middleware - Tool Metering Example")
    print("=" * 60)

    # Step 1: Scrape a website (metered via @meter_tool)
    print("\nStep 1: Scraping website (metered as tool call)...")
    scraped = scrape_website("https://example.com/ai-news")
    print(f"  Scraped: {scraped['title']} ({scraped['word_count']} words)")

    # Step 2: Summarize with GPT (metered via OpenAI middleware)
    print("\nStep 2: Summarizing with GPT (metered as LLM call)...")
    summary = summarize_with_gpt(scraped["content"])
    print(f"  Summary: {summary}")

    print("\n" + "=" * 60)
    print("Both tool calls and LLM calls are metered in Revenium!")
    print("  - Web scraper: tracked via @meter_tool")
    print("  - GPT summary: tracked via OpenAI middleware")
    print("Check your Revenium dashboard to see the usage data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
