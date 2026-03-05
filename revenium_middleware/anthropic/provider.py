"""
Provider detection and configuration for AWS Bedrock support.

This module handles detection of AWS Bedrock vs standard Anthropic based on:
1. Client instance type (boto3 bedrock-runtime client)
2. Base URL substring matching ("amazonaws.com")
3. Default to Anthropic

The detection is simple and focused on the MVP requirements.
"""

import logging
import threading
from enum import Enum, auto
from typing import Optional, Any

logger = logging.getLogger("revenium_middleware.extension")


class Provider(Enum):
    """Supported AI providers."""
    ANTHROPIC = auto()
    BEDROCK = auto()


def detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None) -> Provider:
    """
    Detect which AI provider is being used based on available information.

    Detection priority:
    1. Client instance type (boto3 bedrock-runtime) - most reliable
    2. Base URL substring matching ("amazonaws.com")
    3. Default to Anthropic

    Args:
        client: Client instance (may be boto3 bedrock-runtime client)
        base_url: Base URL for API calls

    Returns:
        Provider enum indicating detected provider
    """
    logger.debug("Detecting AI provider...")

    # 1. Check if client is boto3 bedrock-runtime client (most reliable)
    if client and hasattr(client, "meta"):
        try:
            if hasattr(client.meta, "service_model") and \
               client.meta.service_model.service_name == "bedrock-runtime":
                logger.debug("Bedrock provider detected via boto3 client service_name")
                return Provider.BEDROCK
        except AttributeError:
            # If meta doesn't have service_model, continue to next check
            pass
    
    # 2. Check base URL for AWS substring
    if base_url and "amazonaws.com" in str(base_url).lower():
        logger.debug(f"Bedrock provider detected via base_url: {base_url}")
        return Provider.BEDROCK

    # 3. Check for client base_url if not provided directly
    if client and hasattr(client, 'base_url') and client.base_url:
        if "amazonaws.com" in str(client.base_url).lower():
            logger.debug(f"Bedrock provider detected via client.base_url: {client.base_url}")
            return Provider.BEDROCK

    # 4. Default to Anthropic
    logger.debug("Defaulting to Anthropic provider")
    return Provider.ANTHROPIC


def get_provider_metadata(provider: Provider) -> dict:
    """
    Get provider-specific metadata for usage records.
    
    Args:
        provider: Detected provider
        
    Returns:
        Dictionary with provider and model_source fields
    """
    if provider == Provider.BEDROCK:
        return {
            "provider": "AWS",
            "model_source": "ANTHROPIC"
        }
    else:  # ANTHROPIC
        return {
            "provider": "ANTHROPIC",
            "model_source": "ANTHROPIC"
        }


def is_bedrock_provider(provider: Provider) -> bool:
    """
    Check if the provider is AWS Bedrock.
    
    Args:
        provider: Provider to check
        
    Returns:
        True if AWS Bedrock, False otherwise
    """
    return provider == Provider.BEDROCK


# Thread-local storage for provider cache to ensure thread safety
_thread_local = threading.local()


def _get_thread_cache():
    """Get thread-local cache, initializing if necessary."""
    if not hasattr(_thread_local, 'detected_provider'):
        _thread_local.detected_provider = None
        _thread_local.detection_attempted = False
    return _thread_local


def get_or_detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None,
                          force_redetect: bool = False) -> Provider:
    """
    Get cached provider or detect if not already done.

    This provides lazy loading - detection only happens when needed and is cached
    per thread for thread safety.

    Args:
        client: Client instance
        base_url: Base URL for API calls
        force_redetect: Force re-detection even if cached

    Returns:
        Detected provider
    """
    cache = _get_thread_cache()

    if force_redetect or not cache.detection_attempted:
        cache.detected_provider = detect_provider(client, base_url)
        cache.detection_attempted = True
        logger.debug(f"Provider detection completed: {cache.detected_provider}")

    return cache.detected_provider
