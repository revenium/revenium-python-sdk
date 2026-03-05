# Revenium Python SDK

[![PyPI version](https://img.shields.io/pypi/v/revenium-python-sdk.svg)](https://pypi.org/project/revenium-python-sdk/)
[![Python Versions](https://img.shields.io/pypi/pyversions/revenium-python-sdk.svg)](https://pypi.org/project/revenium-python-sdk/)
[![Documentation](https://img.shields.io/badge/docs-revenium.io-blue)](https://docs.revenium.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official Revenium Python SDK — unified AI metering middleware for deeply attributed AI usage metrics. Supports OpenAI, Anthropic, Google (Gemini/Vertex AI), Ollama, LiteLLM, and Perplexity.

## Features

- **Unified SDK**: Single package with middleware for all major AI providers
- **Asynchronous Processing**: Background thread management for non-blocking metering operations
- **Graceful Shutdown**: Ensures all metering data is properly sent even during application shutdown
- **Decorator Support**: `@revenium_meter` and `@revenium_metadata` for easy integration
- **Tool Metering**: Meter arbitrary tool/function calls alongside LLM API metering

## Supported Providers

| Provider | Extra | Install Command |
|----------|-------|----------------|
| OpenAI | `openai` | `pip install revenium-python-sdk[openai]` |
| Anthropic | `anthropic` | `pip install revenium-python-sdk[anthropic]` |
| Google Gemini | `google-genai` | `pip install revenium-python-sdk[google-genai]` |
| Google Vertex AI | `google-vertex` | `pip install revenium-python-sdk[google-vertex]` |
| Ollama | `ollama` | `pip install revenium-python-sdk[ollama]` |
| LiteLLM | `litellm` | `pip install revenium-python-sdk[litellm]` |
| LiteLLM Proxy | `litellm-proxy` | `pip install revenium-python-sdk[litellm-proxy]` |
| Perplexity (OpenAI) | `perplexity-openai` | `pip install revenium-python-sdk[perplexity-openai]` |
| Perplexity (Native) | `perplexity-native` | `pip install revenium-python-sdk[perplexity-native]` |
| LangChain | `langchain` | `pip install revenium-python-sdk[langchain]` |

## Installation

```bash
# Core SDK
pip install revenium-python-sdk

# With a specific provider
pip install revenium-python-sdk[openai]

# Multiple providers
pip install revenium-python-sdk[openai,anthropic,ollama]
```

## Quick Start

```python
from revenium_middleware import client, run_async_in_thread, shutdown_event

# Record usage directly
client.record_usage(
    model="gpt-4o",
    prompt_tokens=500,
    completion_tokens=200,
    user_id="user123",
    session_id="session456"
)

# Run async metering tasks in background threads
async def async_metering_task():
    await client.async_record_usage(
        model="gpt-3.5-turbo",
        prompt_tokens=300,
        completion_tokens=150,
        user_id="user789"
    )

thread = run_async_in_thread(async_metering_task())

# Application continues while metering happens in background
```

## Provider-Specific Usage

Each provider has its own middleware module. See the `examples/` directory for detailed usage:

- `examples/openai/` — OpenAI and Azure OpenAI examples
- `examples/anthropic/` — Anthropic and Bedrock examples
- `examples/google/` — Google AI and Vertex AI examples
- `examples/ollama/` — Ollama examples
- `examples/litellm/` — LiteLLM client and proxy examples
- `examples/perplexity/` — Perplexity examples

## Tool Metering

The `meter_tool` decorator lets you meter arbitrary tool/function calls (web scrapers, image generators, database lookups, etc.) alongside your LLM API metering. This is available via `revenium_metering` v6.8.2+.

```python
from revenium_middleware import meter_tool, configure

# Configure the metering client
configure(
    metering_url="https://api.revenium.io/meter",
    api_key="your-api-key",
)

# Decorate any tool function to automatically meter it
@meter_tool("my-web-scraper", operation="scrape")
def scrape_website(url):
    # Your scraping logic here
    return {"pages": 5, "data_mb": 2.3}

# The decorator captures timing, success/failure, and reports to Revenium
result = scrape_website("https://example.com")
```

You can also report tool calls manually:

```python
from revenium_middleware import report_tool_call

report_tool_call(
    tool_id="my-tool",
    operation="fetch",
    duration_ms=1234,
    success=True,
    usage_metadata={"records": 42},
)
```

## Compatibility

- Python 3.8+
- Compatible with all supported AI providers

## Logging

This module uses Python's standard logging system. You can control the log level by setting the `REVENIUM_LOG_LEVEL` environment variable:

```bash
# Enable debug logging
export REVENIUM_LOG_LEVEL=DEBUG

# Or when running your script
REVENIUM_LOG_LEVEL=DEBUG python your_script.py
```

Available log levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information (default)
- `WARNING`: Warning messages only
- `ERROR`: Error messages only
- `CRITICAL`: Critical error messages only

## Documentation

For detailed documentation, visit [docs.revenium.io](https://docs.revenium.io)

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

## Code of Conduct

See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

## Security

See [SECURITY.md](./SECURITY.md)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- Built by the Revenium team
