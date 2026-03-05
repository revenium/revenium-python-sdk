"""
Provider detection and configuration for Azure OpenAI support.

This module handles detection of Azure OpenAI vs standard OpenAI based on:
1. Environment variables (AZURE_OPENAI_ENDPOINT)
2. Base URL substring matching ("azure")
3. Client instance type detection

The detection is lazy-loaded and only impacts Azure users.
"""

import os
import logging
import threading
from enum import Enum, auto
from typing import Optional, Any

logger = logging.getLogger("revenium_middleware.extension")


class Provider(Enum):
    """Supported AI providers."""
    OPENAI = auto()
    AZURE_OPENAI = auto()
    OLLAMA = auto()  # Existing provider support


def detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None) -> Provider:
    """
    Detect which AI provider is being used based on available information.

    Detection priority:
    1. Client instance type (AzureOpenAI) - most reliable
    2. Base URL substring matching ("azure")
    3. Environment variables (AZURE_OPENAI_ENDPOINT) - only if client suggests Azure
    4. Default to OpenAI

    Args:
        client: OpenAI client instance (may be AzureOpenAI)
        base_url: Base URL for API calls

    Returns:
        Provider enum indicating detected provider
    """
    logger.debug("Detecting AI provider...")

    # 1. Check client instance type first (most reliable)
    if client and hasattr(client, '__class__'):
        client_class_name = client.__class__.__name__
        if "Azure" in client_class_name:
            logger.debug(f"Azure provider detected via client type: {client_class_name}")
            return Provider.AZURE_OPENAI
    
    # 2. Check base URL for Azure substring (broader than just azure.com)
    if base_url and "azure" in str(base_url).lower():
        logger.debug(f"Azure provider detected via base_url substring: {base_url}")
        return Provider.AZURE_OPENAI

    # 3. Check for client base_url if not provided directly
    if client and hasattr(client, 'base_url') and client.base_url:
        if "azure" in str(client.base_url).lower():
            logger.debug(f"Azure provider detected via client.base_url: {client.base_url}")
            return Provider.AZURE_OPENAI

    # 4. Check for OLLAMA via base URL patterns
    if base_url and ("localhost:11434" in str(base_url) or "ollama" in str(base_url).lower()):
        logger.debug(f"OLLAMA provider detected via base_url: {base_url}")
        return Provider.OLLAMA

    if client and hasattr(client, 'base_url') and client.base_url:
        client_url = str(client.base_url).lower()
        if "localhost:11434" in client_url or "ollama" in client_url:
            logger.debug(f"OLLAMA provider detected via client.base_url: {client.base_url}")
            return Provider.OLLAMA

    # 5. Check environment variables only if we have some indication this might be Azure
    # (This prevents false positives when both Azure and OpenAI configs are present)
    if client and hasattr(client, '__class__') and "Azure" in str(type(client)):
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            logger.debug(f"Azure provider detected via AZURE_OPENAI_ENDPOINT: {azure_endpoint}")
            return Provider.AZURE_OPENAI

    # 6. Default to OpenAI
    logger.debug("Defaulting to OpenAI provider")
    return Provider.OPENAI


def get_provider_metadata(provider: Provider) -> dict:
    """
    Get provider-specific metadata for usage records.
    
    Args:
        provider: Detected provider
        
    Returns:
        Dictionary with provider and model_source fields
    """
    if provider == Provider.AZURE_OPENAI:
        return {
            "provider": "Azure",
            "model_source": "OPENAI"
        }
    elif provider == Provider.OLLAMA:
        return {
            "provider": "OLLAMA", 
            "model_source": "OLLAMA"
        }
    else:  # OPENAI
        return {
            "provider": "OPENAI",
            "model_source": "OPENAI"
        }


def is_azure_provider(provider: Provider) -> bool:
    """
    Check if the provider is Azure OpenAI.
    
    Args:
        provider: Provider to check
        
    Returns:
        True if Azure OpenAI, False otherwise
    """
    return provider == Provider.AZURE_OPENAI


# Global provider cache to avoid repeated detection
_detected_provider: Optional[Provider] = None
_provider_detection_attempted: bool = False
_provider_lock = threading.Lock()  # Thread safety for global state


def get_or_detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None,
                          force_redetect: bool = False) -> Provider:
    """
    Get cached provider or detect if not already done.

    This provides lazy loading - detection only happens when needed and is cached.
    Thread-safe implementation prevents race conditions in multi-threaded environments.

    Args:
        client: OpenAI client instance
        base_url: Base URL for API calls
        force_redetect: Force re-detection even if cached

    Returns:
        Detected provider
    """
    global _detected_provider, _provider_detection_attempted

    # Thread-safe provider detection with double-checked locking pattern
    if force_redetect or not _provider_detection_attempted:
        with _provider_lock:
            # Double-check inside the lock to prevent race conditions
            if force_redetect or not _provider_detection_attempted:
                _detected_provider = detect_provider(client, base_url)
                _provider_detection_attempted = True
                logger.debug(f"Provider detection completed: {_detected_provider}")

    return _detected_provider


def reset_provider_cache():
    """Reset provider detection cache. Useful for testing. Thread-safe implementation."""
    global _detected_provider, _provider_detection_attempted
    with _provider_lock:
        _detected_provider = None
        _provider_detection_attempted = False
