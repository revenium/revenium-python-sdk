"""
Provider detection and configuration for AWS Bedrock and Microsoft Foundry support.

This module handles detection of AWS Bedrock and Microsoft Foundry vs standard
Anthropic based on:
1. Client instance type (boto3 bedrock-runtime client, Anthropic Foundry client)
2. Base URL substring matching ("amazonaws.com", "services.ai.azure.com")
3. Default to Anthropic

The detection is simple and focused on the MVP requirements.
"""

import logging
import os
import threading
from enum import Enum, auto
from typing import Optional, Any

logger = logging.getLogger("revenium_middleware.extension")

# Foundry clients are the Anthropic SDK's own AnthropicFoundry /
# AsyncAnthropicFoundry (both deriving from BaseFoundryClient), so a name
# marker on the class hierarchy identifies them without importing anthropic
# here -- this module must stay usable when the SDK is not installed.
_FOUNDRY_CLASS_MARKER = "foundry"

# The name marker only counts on classes the Anthropic SDK itself defines.
# Without that constraint a third-party wrapper -- say a FoundryProxyClient
# that forwards to a plain Anthropic client -- would be mislabelled as Foundry
# purely because of how it is named.
_FOUNDRY_CLASS_MODULE = "anthropic"

# Default Foundry hosts look like https://<resource>.services.ai.azure.com/anthropic/,
# but the resource name is customer-specific and base_url can be overridden
# entirely, so the host shape is only a secondary signal.
_FOUNDRY_HOST_MARKER = "services.ai.azure.com"


class Provider(Enum):
    """Supported AI providers."""
    ANTHROPIC = auto()
    BEDROCK = auto()
    FOUNDRY = auto()


def _is_anthropic_sdk_class(klass: Any) -> bool:
    """
    Check whether a class was defined inside the Anthropic SDK package.

    The dotted boundary is deliberate: a plain prefix test would also accept a
    third-party module such as "anthropic_foundry_proxy".

    Args:
        klass: Class to inspect

    Returns:
        True if the class comes from the anthropic package
    """
    module = getattr(klass, "__module__", "") or ""
    return module == _FOUNDRY_CLASS_MODULE or module.startswith(f"{_FOUNDRY_CLASS_MODULE}.")


def _is_foundry_client(client: Optional[Any]) -> bool:
    """
    Check whether a client instance is an Anthropic Foundry client.

    The whole class hierarchy is inspected so subclasses of AnthropicFoundry
    and AsyncAnthropicFoundry are recognised too. A class only counts when the
    Anthropic SDK defines it, so a caller's own Foundry-named wrapper around a
    direct Anthropic client stays attributed to Anthropic.

    Args:
        client: Client instance to inspect

    Returns:
        True if the client is (or derives from) a Foundry client
    """
    if client is None:
        return False

    client_type = type(client)
    for klass in getattr(client_type, "__mro__", (client_type,)):
        if not _is_anthropic_sdk_class(klass):
            continue
        if _FOUNDRY_CLASS_MARKER in getattr(klass, "__name__", "").lower():
            return True

    return False


def _detect_foundry_provider(client: Optional[Any] = None,
                             base_url: Optional[str] = None) -> Optional[Provider]:
    """
    Detect Microsoft Foundry from the client class or an Azure Foundry host.

    The client class is checked first because a Foundry client may be pointed
    at a custom base_url, in which case the host shape says nothing.

    Args:
        client: Client instance (may be an Anthropic Foundry client)
        base_url: Base URL for API calls

    Returns:
        Provider.FOUNDRY when a Foundry signal is found, otherwise None
    """
    if _is_foundry_client(client):
        logger.debug(f"Foundry provider detected via client type: {type(client).__name__}")
        return Provider.FOUNDRY

    client_base_url = getattr(client, "base_url", None) if client else None
    for candidate in (base_url, client_base_url):
        if candidate and _FOUNDRY_HOST_MARKER in str(candidate).lower():
            logger.debug(f"Foundry provider detected via base_url: {candidate}")
            return Provider.FOUNDRY

    return None


def detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None) -> Provider:
    """
    Detect which AI provider is being used based on available information.

    Detection priority:
    1. Client instance type (boto3 bedrock-runtime) - most reliable
    2. Base URL substring matching ("amazonaws.com")
    3. Client instance type (Anthropic Foundry client) - most reliable
    4. Base URL substring matching ("services.ai.azure.com")
    5. Default to Anthropic

    REVENIUM_BEDROCK_DISABLE=1 switches off the Bedrock signals (1-2) only.
    Foundry detection still runs, so a Foundry client is never mislabelled as
    direct Anthropic just because Bedrock routing was turned off - the switch
    is documented as disabling Bedrock support, nothing else. Callers rely on
    this single gate rather than re-checking the variable themselves.

    Args:
        client: Client instance (may be boto3 bedrock-runtime client or an
            Anthropic Foundry client)
        base_url: Base URL for API calls

    Returns:
        Provider enum indicating detected provider
    """
    logger.debug("Detecting AI provider...")

    if os.getenv("REVENIUM_BEDROCK_DISABLE") == "1":
        logger.debug("Bedrock detection disabled via REVENIUM_BEDROCK_DISABLE")
    else:
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

    # 4. Check for a Microsoft Foundry client or an Azure Foundry host
    foundry_provider = _detect_foundry_provider(client=client, base_url=base_url)
    if foundry_provider is not None:
        return foundry_provider

    # 5. Default to Anthropic
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
    elif provider == Provider.FOUNDRY:
        # Foundry gets its own vendor bucket rather than reusing "Azure": the
        # Azure label belongs to the platform's regional-pricing providers, so
        # it would stack a regional surcharge on top of the Anthropic rate card
        # that Foundry actually bills at. It also collides with the uppercase
        # convention the enforcement rules reserve. model_source stays
        # ANTHROPIC because Foundry serves Anthropic's own models.
        return {
            "provider": "Foundry",
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
