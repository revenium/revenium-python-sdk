"""
Prompt extraction utilities for capturing AI prompts and responses.

The OpenAI-shaped extractors live in ``revenium_middleware._core.prompt_extraction``
so the LiteLLM integration can use them without importing this package, which
patches the OpenAI SDK. They are re-exported here under their original names.
"""

from revenium_middleware._core.prompt_extraction import (  # noqa: F401 (re-exported)
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content,
)
