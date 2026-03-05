"""
Revenium Middleware for Perplexity AI

This library automatically tracks Perplexity AI API usage and sends
metering data to Revenium. Simply import this module and all Perplexity
API calls will be automatically tracked.

Supports both OpenAI SDK and native Perplexity SDK:

Example 1 - OpenAI SDK:
    import revenium_middleware.perplexity
    from openai import OpenAI

    client = OpenAI(
        api_key="your-perplexity-api-key",
        base_url="https://api.perplexity.ai"
    )

    response = client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Example 2 - Native Perplexity SDK:
    import revenium_middleware.perplexity
    from perplexity import Perplexity

    client = Perplexity(api_key="your-perplexity-api-key")

    response = client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Example 3 - With Decorators:
    from revenium_middleware.perplexity import revenium_metadata
    from openai import OpenAI

    client = OpenAI(
        api_key="your-perplexity-api-key",
        base_url="https://api.perplexity.ai"
    )

    @revenium_metadata(
        organization_id="acme-corp",
        task_type="analysis"
    )
    def analyze_text(text):
        # Metadata automatically injected!
        response = client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": text}]
        )
        return response.choices[0].message.content
"""
import logging

logger = logging.getLogger("revenium_middleware.perplexity")

# Conditionally import middleware (requires wrapt)
try:
    import wrapt  # noqa: F401
    from .middleware import create_wrapper
    from .perplexity_sdk import perplexity_create_wrapper
except ImportError:
    logger.debug("Perplexity middleware dependencies not available, middleware not loaded")
    create_wrapper = None  # type: ignore
    perplexity_create_wrapper = None  # type: ignore

# Re-export decorators from revenium_middleware for convenience
try:
    from revenium_middleware import revenium_metadata, revenium_meter
    _decorators_available = True
except ImportError:
    _decorators_available = False
    revenium_metadata = None  # type: ignore
    revenium_meter = None  # type: ignore

# Both wrappers are automatically applied when this module is imported
# via the @wrapt.patch_function_wrapper decorators

__version__ = "0.1.0"

if _decorators_available:
    __all__ = [
        "create_wrapper",
        "perplexity_create_wrapper",
        "revenium_metadata",
        "revenium_meter"
    ]
else:
    __all__ = ["create_wrapper", "perplexity_create_wrapper"]

logger.debug(
    "Revenium Perplexity middleware loaded - "
    "both OpenAI and Perplexity SDK wrappers active"
)
