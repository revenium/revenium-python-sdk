"""
Provider detection for Perplexity AI API.

This module handles detection of Perplexity as the AI provider
and provides metadata about the provider.
"""
import logging
from enum import Enum
from typing import Optional, Any, Dict

logger = logging.getLogger("revenium_middleware.perplexity.provider")


class Provider(str, Enum):
    """Supported AI providers."""
    PERPLEXITY = "PERPLEXITY"
    OPENAI = "OPENAI"  # Fallback for OpenAI-compatible APIs


def detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None) -> Provider:
    """
    Detect which AI provider is being used based on available information.

    Detection priority:
    1. Base URL substring matching ("perplexity.ai")
    2. Client base_url attribute
    3. Default to PERPLEXITY (since this is Perplexity middleware)

    Args:
        client: OpenAI client instance
        base_url: Base URL for API calls

    Returns:
        Provider enum indicating detected provider
    """
    logger.debug("Detecting AI provider...")

    # 1. Check base URL for Perplexity substring
    if base_url and "perplexity" in str(base_url).lower():
        logger.debug(f"Perplexity provider detected via base_url: {base_url}")
        return Provider.PERPLEXITY

    # 2. Check for client base_url if not provided directly
    if client and hasattr(client, 'base_url') and client.base_url:
        if "perplexity" in str(client.base_url).lower():
            logger.debug(f"Perplexity provider detected via client.base_url: {client.base_url}")
            return Provider.PERPLEXITY

    # 3. Default to Perplexity (this is Perplexity middleware)
    logger.debug("Defaulting to Perplexity provider")
    return Provider.PERPLEXITY


def get_provider_metadata(provider: Provider) -> Dict[str, str]:
    """
    Get metadata for the detected provider.

    Args:
        provider: Provider enum value

    Returns:
        Dictionary with provider metadata
    """
    metadata = {
        "provider": provider.value,
        "model_source": provider.value,
    }
    
    logger.debug(f"Provider metadata: {metadata}")
    return metadata


def is_perplexity_provider(provider: Provider) -> bool:
    """
    Check if the provider is Perplexity.

    Args:
        provider: Provider enum value

    Returns:
        True if provider is Perplexity, False otherwise
    """
    return provider == Provider.PERPLEXITY

